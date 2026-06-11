"""Episode → SLAM (→ LeRobot → Hub) pipeline, run in-process inside the Space.

Thin glue over grabette_postprocess. The SLAM binary is bundled in the image
(no Docker), so run_oak_slam is called with binary=OAK_VSLAM_BINARY.

Test mode (current): run_slam_pipeline stops after SLAM and returns per-episode
trajectories for visualization. The full build+push path (process_dataset) is
kept for later — see the angle_data.json caveat before re-enabling it.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render plots to images, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grabette_postprocess.convert import convert_episode
from grabette_postprocess.oak_slam import run_oak_slam

BINARY = os.environ.get("OAK_VSLAM_BINARY", "/usr/local/bin/offline_vslam")


def find_episode_dirs(root: Path) -> list[Path]:
    """Every directory containing a raw OAK-D recording, anywhere under root.

    Recursive so it works whether the zip wraps episodes in a folder, puts them
    at the top level, or contains a single episode at the root.
    """
    return sorted({p.parent for p in Path(root).rglob("oakd_left.mp4")})


def run_slam_pipeline(dataset_dir, log=print) -> list[dict]:
    """Convert + SLAM every episode. Stops at SLAM (no LeRobot, no push).

    Returns one dict per episode that produced a trajectory:
        {name, traj_path, tracking_pct, tracked, total}
    """
    episodes = find_episode_dirs(Path(dataset_dir))
    if not episodes:
        raise ValueError(f"No episodes (oakd_left.mp4) found under {dataset_dir}")

    log(f"Found {len(episodes)} episode(s)")
    results = []
    for ep in episodes:
        log(f"▶ {ep.name}: convert")
        convert_episode(ep)

        log(f"▶ {ep.name}: SLAM")
        r = run_oak_slam(ep, binary=BINARY, show_progress=False)
        if r.trajectory_path is None:
            log(f"  ✗ {ep.name}: SLAM failed (rc={r.returncode})")
            continue

        log(f"  ✓ {ep.name}: tracking {r.tracking_pct:.1f}% ({r.tracked_frames}/{r.total_frames})")
        results.append({
            "name": ep.name,
            "traj_path": r.trajectory_path,
            "tracking_pct": r.tracking_pct,
            "tracked": r.tracked_frames,
            "total": r.total_frames,
        })
    return results


def plot_trajectory(traj_path: Path, title: str) -> np.ndarray:
    """Render a gravity-aligned trajectory (Z-up) as a 3D plot image (RGBA)."""
    df = pd.read_csv(traj_path)
    xyz = df[["x", "y", "z"]].to_numpy()
    lost = df["is_lost"].astype(bool).to_numpy()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.75", lw=1)
    ax.scatter(*xyz[~lost].T, c="tab:blue", s=4, label="tracked")
    if lost.any():
        ax.scatter(*xyz[lost].T, c="tab:red", s=4, label="lost")
    ax.scatter(*xyz[0], c="green", s=50, marker="o", label="start")

    # Equal aspect so the path isn't distorted.
    ranges = xyz.max(axis=0) - xyz.min(axis=0)
    ax.set_box_aspect(np.where(ranges > 0, ranges, 1))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z (up) [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return img


# --- Full build+push path, kept for later (NOT used in SLAM test mode) ---------
# ⚠ Before re-enabling: build_dataset reads imu_data.json (GoPro schema), but
# joint angles are recorded in angle_data.json (flat schema). interpolate_angles
# must be adapted, otherwise joints come out as zeros.

def process_dataset(dataset_dir, target_repo, task, root, log=print) -> int:
    """Convert + SLAM + build LeRobot dataset + push to Hub. Caller must
    huggingface_hub.login() first so push_to_hub uses that account."""
    from grabette_postprocess.dataset import build_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    results = run_slam_pipeline(dataset_dir, log=log)
    if not results:
        raise RuntimeError("No episode produced a trajectory; nothing to push.")
    processed = [Path(r["traj_path"]).parent for r in results]

    log(f"Building LeRobot dataset from {len(processed)} episode(s)…")
    build_dataset(repo_id=target_repo, episode_dirs=processed, task=task, root=Path(root))

    log(f"Pushing to https://huggingface.co/datasets/{target_repo} …")
    ds = LeRobotDataset(target_repo, root=Path(root))
    ds.push_to_hub(tags=["lerobot", "grabette"])
    return len(processed)
