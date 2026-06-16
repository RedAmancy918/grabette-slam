"""Build a Rerun recording of an episode's SLAM trajectory + camera frame.

Used to drop a ``trajectory.rrd`` next to each episode at the end of the local
SLAM run, so the trajectory can be inspected later in the Rerun viewer
(``rerun <episode>/trajectory.rrd``) without re-running anything.

The 3D content mirrors scripts/rgbd_slam/visualize_rgbd_trajectory.py (its
default ``--no-gravity-align`` path): a gravity-aligned Z-up world, the full
trajectory (green), start/end markers, an animated camera pose with RGB body
axes ("le repère de la caméra"), and the world-frame gravity arrow. The
trajectory is assumed already gravity-aligned — ``run_oak_slam`` writes
camera_trajectory.csv that way — so we log it as-is.

Saving to a ``.rrd`` (rather than spawning a viewer) is what makes this usable
from a headless / batch context; the standalone script remains the way to get a
live, interactive viewer with video + IMU.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _gravity_world(oak_dir: Path, quat0: np.ndarray) -> np.ndarray | None:
    """Gravity direction in the (already gravity-aligned) world frame.

    Should be ~(0, 0, -1) by construction, but we recompute it from the IMU so
    a tilted result is a visible red flag that the alignment went wrong. Returns
    None when the oak/ sidecar files aren't available.
    """
    try:
        from scipy.spatial.transform import Rotation

        imu_acc = pd.read_csv(oak_dir / "imu_acc.csv")
        g_imu = imu_acc[["ax", "ay", "az"]].iloc[:50].mean().to_numpy()
        calib = json.loads((oak_dir / "calib_offline.json").read_text())
        R_imu_to_cam = np.array(calib["imu_to_cam"])[:3, :3]
        # Accel at rest is specific force (UP); negate for physical gravity (DOWN).
        g_cam = -(R_imu_to_cam @ g_imu)
        g_cam /= np.linalg.norm(g_cam)
        return Rotation.from_quat(quat0).apply(g_cam)
    except Exception:
        return None


def log_episode_trajectory(
    episode_dir: Path | str,
    *,
    traj_csv_name: str = "camera_trajectory.csv",
    video_skip: int = 5,
    with_video: bool = False,
) -> bool:
    """Log one episode's trajectory + camera frame to the *current* Rerun recording.

    Caller is responsible for ``rr.init``/``rr.save``/``rr.spawn`` around this.
    Returns True if a trajectory was logged, False if there was nothing to show.
    """
    import rerun as rr
    import rerun.blueprint as rrb
    from scipy.spatial.transform import Rotation

    episode_dir = Path(episode_dir)
    oak_dir = episode_dir / "oak"

    traj_csv = episode_dir / traj_csv_name
    if not traj_csv.is_file():
        return False

    df_all = pd.read_csv(traj_csv)
    df_valid = df_all[~df_all["is_lost"].astype(bool)].copy()
    if df_valid.empty:
        return False

    positions = df_valid[["x", "y", "z"]].to_numpy()
    quaternions = df_valid[["q_x", "q_y", "q_z", "q_w"]].to_numpy()

    # Normalize timestamps so the Rerun timeline starts at 0.
    t0 = float(df_all["timestamp"].iloc[0])
    df_all = df_all.copy()
    df_all["timestamp"] = df_all["timestamp"] - t0

    # Gravity-aligned world: Z-up, gravity = -Z.
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    video_cap = None
    cv2 = None
    if with_video:
        import cv2

        video_path = episode_dir / "oakd_left.mp4"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            video_cap = cap if cap.isOpened() else None

    # Blueprint: 3D view (+ camera feed panel when video is included).
    try:
        views = [rrb.Spatial3DView(name="3D View", origin="/world")]
        if video_cap is not None:
            views.append(rrb.Spatial2DView(name="Camera", origin="/camera_feed"))
        rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views)))
    except Exception:
        pass

    # --- Static elements ---
    axis_len = 0.5
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ), static=True)
    rr.log("world/trajectory_full", rr.LineStrips3D(positions, colors=[0, 255, 0]), static=True)
    rr.log("world/start", rr.Points3D(positions[0], colors=[0, 255, 0], radii=0.01), static=True)
    rr.log("world/end", rr.Points3D(positions[-1], colors=[255, 0, 0], radii=0.01), static=True)

    g_world = _gravity_world(oak_dir, quaternions[0]) if oak_dir.is_dir() else None
    if g_world is not None:
        rr.log("world/gravity", rr.Arrows3D(
            origins=[positions[0].tolist()],
            vectors=[(g_world * 0.3).tolist()],
            colors=[[255, 255, 0]],
            labels=["g (world)"],
        ), static=True)

    # --- Animated camera frame along the trajectory ---
    pose_map = {
        int(row["frame_idx"]): (
            np.array([row["x"], row["y"], row["z"]]),
            np.array([row["q_x"], row["q_y"], row["q_z"], row["q_w"]]),
        )
        for _, row in df_valid.iterrows()
    }

    cam_axis_len = 0.1
    history: list[np.ndarray] = []
    for frame_i in range(len(df_all)):
        if frame_i % video_skip != 0:
            continue
        row = df_all.iloc[frame_i]
        frame_idx = int(row["frame_idx"])
        rr.set_time("time", timestamp=float(row["timestamp"]))

        if frame_idx in pose_map:
            pos, quat = pose_map[frame_idx]
            rr.log("world/camera", rr.Transform3D(translation=pos.tolist(), quaternion=quat.tolist()))
            rr.log("world/current_position", rr.Points3D(pos, colors=[255, 0, 0], radii=0.005))

            history.append(pos)
            if len(history) > 1:
                rr.log("world/trajectory_history",
                       rr.LineStrips3D(np.array(history), colors=[0, 128, 255]))

            rot = Rotation.from_quat(quat)
            rr.log("world/camera_axes", rr.Arrows3D(
                origins=[pos, pos, pos],
                vectors=[rot.apply([cam_axis_len, 0, 0]),
                         rot.apply([0, cam_axis_len, 0]),
                         rot.apply([0, 0, cam_axis_len])],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ))

        if video_cap is not None:
            # oakd_left.mp4 frames are 1:1 with the SLAM frame_idx.
            video_cap.set(1, frame_idx)  # cv2.CAP_PROP_POS_FRAMES
            ret, frame = video_cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 400), interpolation=cv2.INTER_AREA)
                rr.log("camera_feed", rr.Image(frame))

    if video_cap is not None:
        video_cap.release()
    return True


def save_episode_rrd(
    episode_dir: Path | str,
    rrd_path: Path | str | None = None,
    *,
    traj_csv_name: str = "camera_trajectory.csv",
    app_id: str = "grabette_traj",
    **kwargs,
) -> Path | None:
    """Build a Rerun recording for one episode and write it to a ``.rrd``.

    ``rrd_path`` defaults to ``<episode_dir>/trajectory.rrd``. Returns the path
    written, or None if the episode had no usable trajectory. Extra kwargs
    (``video_skip``, ``with_video``) pass through to log_episode_trajectory.
    """
    import rerun as rr

    episode_dir = Path(episode_dir)
    rrd_path = Path(rrd_path) if rrd_path is not None else episode_dir / "trajectory.rrd"

    # A fresh, non-global recording so this never disturbs an active viewer and
    # is safe to call repeatedly in a batch loop.
    rec = rr.RecordingStream(application_id=app_id, recording_id=episode_dir.name)
    with rec:
        logged = log_episode_trajectory(
            episode_dir, traj_csv_name=traj_csv_name, **kwargs
        )
        if not logged:
            return None
        rr.save(str(rrd_path), recording=rec)
    return rrd_path
