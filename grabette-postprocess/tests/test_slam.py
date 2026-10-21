"""Tests for the IMU-less fallback of the world-frame alignment
(grabette_postprocess.slam._gravity_align_trajectory). No SLAM binary needed.
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from grabette_postprocess.slam import R_FLU_FROM_ODOM, _gravity_align_trajectory


def _traj():
    return pd.DataFrame({
        "is_lost": [0, 0], "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 1.0],
        "q_x": [0.0, 0.0], "q_y": [0.0, 0.0], "q_z": [0.0, 0.0], "q_w": [1.0, 1.0],
    })


def test_imu_less_camera_logs_plainly_and_uses_fixed_rotation(tmp_path, capsys):
    # A Gemini 305's slam_input/ has a calib but no IMU CSVs.
    (tmp_path / "calib_offline.json").write_text("{}")
    out = _gravity_align_trajectory(_traj(), tmp_path)
    log = capsys.readouterr().out
    assert "IMU-less camera" in log
    assert "Errno" not in log
    expected = Rotation.from_matrix(R_FLU_FROM_ODOM).apply([0.0, 0.0, 1.0])
    np.testing.assert_allclose(out[["x", "y", "z"]].iloc[1], expected, atol=1e-9)


def test_broken_imu_data_keeps_the_error(tmp_path, capsys):
    # IMU CSV present but the calib lacks imu_to_cam: a real problem, so the
    # underlying error stays in the log.
    pd.DataFrame({"timestamp": [0.0], "ax": [0.0], "ay": [0.0], "az": [9.81]}).to_csv(
        tmp_path / "imu_acc.csv", index=False)
    (tmp_path / "calib_offline.json").write_text("{}")
    _gravity_align_trajectory(_traj(), tmp_path)
    log = capsys.readouterr().out
    assert "no IMU gravity" in log and "imu_to_cam" in log
