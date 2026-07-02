"""Derive LeRobot-compatible per-episode tags from the pipeline's checks.

Tags are short machine-readable labels written into the built dataset's
per-episode metadata (a `tags` list-of-strings column in meta/episodes/*.parquet;
see dataset._write_episode_tags). LeRobot preserves extra episode-metadata columns
and exposes them as ``ds.meta.episodes[i]["tags"]``, so downstream training can
filter episodes by a known quality issue without re-running the checks.

Two sources, mirroring the two check phases:
  - recording (pre-SLAM)  → TAG_FIXED_GRIPPER  (a gripper joint never actuated)
  - trajectory (post-SLAM) → TAG_TRACKING_LOST (SLAM lost the camera)

Auto-derived only — there is no manual tagging path.
"""

from pathlib import Path

from grabette_postprocess.checks.recording import static_gripper_joints

TAG_FIXED_GRIPPER = "fixed_gripper"
TAG_TRACKING_LOST = "tracking_lost"

# Tracked-frame percentage below which SLAM effectively lost the camera over much
# of the episode (matches the "Low tracking" trajectory warning threshold).
TRACKING_LOST_PCT = 50.0


def recording_tags(ep_dir: Path) -> list[str]:
    """Tags derivable from the raw recording (pre-SLAM). Global `fixed_gripper`
    when either gripper joint stayed static over the whole episode."""
    return [TAG_FIXED_GRIPPER] if static_gripper_joints(ep_dir) else []


def trajectory_tags(report) -> list[str]:
    """Tags derivable from the SLAM trajectory report (post-SLAM). `tracking_lost`
    when tracking was essentially absent (< 2 tracked frames) or SLAM held the
    camera for less than TRACKING_LOST_PCT of the episode. None-safe."""
    if report is None:
        return []
    if report.n_tracked < 2 or report.tracking_pct < TRACKING_LOST_PCT:
        return [TAG_TRACKING_LOST]
    return []


def episode_tags(ep_dir: Path, report=None) -> list[str]:
    """Combined, de-duplicated, stable-ordered tags for one built episode.

    ep_dir: the raw episode directory (for the recording-derived tags).
    report: the episode's TrajectoryReport, or None when unavailable.
    """
    seen: list[str] = []
    for tag in (*recording_tags(ep_dir), *trajectory_tags(report)):
        if tag not in seen:
            seen.append(tag)
    return seen
