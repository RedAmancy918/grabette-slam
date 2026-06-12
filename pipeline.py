"""Episode → SLAM → LeRobot → Hub pipeline, run in-process inside the Space.

Thin glue over grabette_postprocess. The SLAM binary is bundled in the image
(no Docker), so run_oak_slam is called with binary=OAK_VSLAM_BINARY.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grabette_postprocess.convert import convert_episode
from grabette_postprocess.oak_slam import run_oak_slam
from grabette_postprocess.dataset import build_dataset
from grabette_postprocess.trajectory import analyze_trajectory
from grabette_postprocess.episode_check import check_episode

BINARY = os.environ.get("OAK_VSLAM_BINARY", "/usr/local/bin/offline_vslam")


def find_episode_dirs(root: Path) -> list[Path]:
    """Every directory containing a raw OAK-D recording, anywhere under root.

    Recursive so it works whether the dataset wraps episodes in a folder, puts
    them at the top level, or is a single episode at the root.
    """
    return sorted({p.parent for p in Path(root).rglob("oakd_left.mp4")})


def _process_episode(ep: Path) -> tuple[Path | None, list[str]]:
    """Convert + SLAM + quality-check one episode (runs in a worker thread).

    Returns (episode_dir | None, log_lines). The dir is None when SLAM produced
    no trajectory (the episode is dropped). Log lines are collected rather than
    emitted directly so concurrent episodes don't interleave in the output.
    The trajectory quality check is advisory: a BAD/WARN verdict is logged but
    the episode is still kept.
    """
    # Advisory input check — validates the raw recording (Arducam / OAK RGBD+IMU
    # / gripper angles) before SLAM. Logged, never blocks (a noisy episode can
    # still produce a usable trajectory).
    lines = []
    chk = check_episode(ep)
    if chk["errors"] or chk["warnings"]:
        verdict = "ERROR" if chk["errors"] else "WARN"
        lines.append(f"  [input/{verdict}] {ep.name}")
        for msg in (*chk["errors"], *chk["warnings"]):
            lines.append(f"      • {msg}")

    lines.append(f"▶ {ep.name}: convert")
    convert_episode(ep)

    lines.append(f"▶ {ep.name}: SLAM")
    r = run_oak_slam(ep, binary=BINARY, show_progress=False)
    if r.trajectory_path is None:
        lines.append(f"  ✗ {ep.name}: SLAM failed (rc={r.returncode})")
        return None, lines
    lines.append(f"  ✓ {ep.name}: tracking {r.tracking_pct:.1f}% ({r.tracked_frames}/{r.total_frames})")

    # Advisory trajectory quality check — flags drift/jumps/zigzag but never drops.
    report = analyze_trajectory(r.trajectory_path, ep / "slam_metadata.json")
    lines.append(
        f"  [{report.verdict}] {ep.name}: dist={report.total_distance_m:.2f}m "
        f"med_step={report.median_step_mm:.1f}mm jumps={report.n_jumps}"
    )
    for msg in (*report.errors, *report.warnings):
        lines.append(f"      • {msg}")
    return ep, lines


def run_slam(dataset_dir, log=print) -> list[Path]:
    """Convert + SLAM every episode in parallel. Returns dirs that produced a trajectory.

    Episodes are independent (each reads/writes only its own dir), so they run
    concurrently. SLAM is a single-threaded CPU-bound subprocess, so workers are
    capped at the core count; processing whole episodes per worker also overlaps
    one episode's convert with another's SLAM for free.
    """
    episodes = find_episode_dirs(Path(dataset_dir))
    if not episodes:
        raise ValueError(f"No episodes (oakd_left.mp4) found under {dataset_dir}")

    log(f"Found {len(episodes)} episode(s)")
    workers = min(len(episodes), os.cpu_count() or 2)
    processed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_episode, ep) for ep in episodes]
        for fut in as_completed(futures):
            ep, lines = fut.result()
            for line in lines:
                log(line)
            if ep is not None:
                processed.append(ep)
    return sorted(processed)


def process_dataset(dataset_dir, target_repo, task, root, log=print) -> int:
    """Convert + SLAM + build a LeRobot dataset + push it to the Hub.

    The caller must set the HF_TOKEN env var (with write/manage-repos scope) first
    so push_to_hub uploads under that account. Returns the episode count.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    processed = run_slam(dataset_dir, log=log)
    if not processed:
        raise RuntimeError("No episode produced a trajectory; nothing to push.")

    log(f"Building LeRobot dataset from {len(processed)} episode(s)…")
    build_dataset(repo_id=target_repo, episode_dirs=processed, task=task, root=Path(root))

    log(f"Pushing to https://huggingface.co/datasets/{target_repo} …")
    ds = LeRobotDataset(target_repo, root=Path(root))
    ds.push_to_hub(tags=["lerobot", "grabette"])
    return len(processed)
