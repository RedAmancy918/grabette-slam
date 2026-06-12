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


def run_slam(dataset_dir, log=print, should_stop=None, on_progress=None) -> list[Path]:
    """Convert + SLAM every episode in parallel. Returns dirs that produced a trajectory.

    Episodes are independent (each reads/writes only its own dir), so they run
    concurrently. SLAM is a single-threaded CPU-bound subprocess, so workers are
    capped at the core count; processing whole episodes per worker also overlaps
    one episode's convert with another's SLAM for free.

    should_stop: optional no-arg predicate; when it returns True the loop stops
    collecting results and cancels not-yet-started episodes (a running SLAM
    finishes on its own — a subprocess can't be interrupted mid-call).
    on_progress: optional callback(done, total, "slam") fired per finished episode.
    """
    episodes = find_episode_dirs(Path(dataset_dir))
    if not episodes:
        raise ValueError(f"No episodes (oakd_left.mp4) found under {dataset_dir}")

    total = len(episodes)
    log(f"Found {total} episode(s)")
    if on_progress:
        on_progress(0, total, "slam")
    workers = min(total, os.cpu_count() or 2)
    processed = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_episode, ep) for ep in episodes]
        for fut in as_completed(futures):
            if should_stop and should_stop():
                log("⛔ Stop requested — abandoning remaining episodes")
                pool.shutdown(wait=False, cancel_futures=True)
                break
            ep, lines = fut.result()
            for line in lines:
                log(line)
            if ep is not None:
                processed.append(ep)
            done += 1
            if on_progress:
                on_progress(done, total, "slam")
    return sorted(processed)


def process_dataset(dataset_dir, target_repo, task, root, log=print,
                    should_stop=None, create_pr=False, on_progress=None,
                    token=None) -> tuple[int, str | None, str]:
    """Convert + SLAM + build a LeRobot dataset, then push it (or open a PR).

    token: HF token used for the upload. Also exported as HF_TOKEN (so the
        push_to_hub path, which builds its own HfApi internally, picks it up).

    create_pr: if True, upload the built dataset as a Pull Request on an existing
        target repo (HfApi.upload_folder(create_pr=True)) instead of committing to
        main via push_to_hub. Opening a PR needs write access to the repo — a
        read-only token returns 403.
    should_stop: optional no-arg predicate; checked before the (irreversible)
        build + push so a stop request never publishes a partial dataset.
    on_progress: optional callback(done, total, phase) for UI progress.

    Returns (episode_count, link_or_None, mode) where mode is one of
    "main" (pushed to main), "pr" (Pull Request opened), or "branch" (pushed to
    a branch as a PR fallback). link is the PR/branch URL when relevant.
    """
    import os
    import re

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from huggingface_hub import HfApi

    if token:
        os.environ["HF_TOKEN"] = token  # picked up by push_to_hub's internal HfApi

    processed = run_slam(dataset_dir, log=log, should_stop=should_stop, on_progress=on_progress)
    if should_stop and should_stop():
        raise RuntimeError("Stopped by user — nothing pushed.")
    if not processed:
        raise RuntimeError("No episode produced a trajectory; nothing to push.")

    if on_progress:
        on_progress(0, 0, "build")
    log(f"Building LeRobot dataset from {len(processed)} episode(s)…")
    build_dataset(repo_id=target_repo, episode_dirs=processed, task=task, root=Path(root))
    ds = LeRobotDataset(target_repo, root=Path(root))

    if on_progress:
        on_progress(0, 0, "push")
    if create_pr:
        # 1) Try a real PR. HF OAuth ("Sign in with HF") tokens are often refused
        #    by the PR/discussion endpoint even with write-repos, so this may 403.
        try:
            log(f"Opening a Pull Request on {target_repo} …")
            info = HfApi(token=token).upload_folder(
                repo_id=target_repo, repo_type="dataset", folder_path=str(ds.root),
                ignore_patterns=["images/"], create_pr=True,
                commit_message=f"Add grabette SLAM dataset: {task}",
            )
            pr_url = getattr(info, "pr_url", None)
            log(f"✅ PR opened: {pr_url}")
            return len(processed), pr_url, "pr"
        except Exception as e:
            log(f"  PR rejected ({e}); falling back to pushing a branch…")

        # 2) Fallback: push to a dedicated branch (a plain commit — write-repos
        #    allows this even when opening a PR directly is not permitted).
        slug = re.sub(r"[^a-z0-9]+", "-", task.lower()).strip("-")[:40] or "update"
        branch = f"grabette-{slug}"
        try:
            # tag_version=False: don't create the repo-global "v3.0" version tag
            # when we're only proposing a branch on someone's existing repo.
            ds.push_to_hub(branch=branch, tags=["lerobot", "grabette"], tag_version=False)
        except Exception as e:
            raise RuntimeError(
                f"Could not write to {target_repo} (HF said: {e}). Your sign-in token "
                f"appears read-only — re-login to grant write-repos, or run the Space "
                f"with a write token."
            ) from e
        branch_url = f"https://huggingface.co/datasets/{target_repo}/tree/{branch}"
        log(f"✅ Pushed to branch '{branch}'.")
        return len(processed), branch_url, "branch"

    log(f"Pushing to https://huggingface.co/datasets/{target_repo} …")
    ds.push_to_hub(tags=["lerobot", "grabette"])
    return len(processed), None, "main"
