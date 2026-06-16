#!/usr/bin/env python3
"""Run OAK-D SR offline VSLAM on one or more episode directories."""

import click
from pathlib import Path

from grabette_postprocess.oak_slam import run_oak_slam, DEFAULT_DOCKER_IMAGE


@click.command()
@click.option("-i", "--input_dir", required=True, multiple=True,
              type=click.Path(exists=True),
              help="Episode directory containing oak/ (repeatable for batch)")
@click.option("-d", "--docker_image", default=DEFAULT_DOCKER_IMAGE, show_default=True,
              help="Docker image name")
@click.option("--output_csv", default="camera_trajectory.csv", show_default=True,
              help="Output trajectory filename inside each episode directory")
@click.option("--save-rrd/--no-save-rrd", default=True, show_default=True,
              help="Write a Rerun trajectory.rrd (3D trajectory + camera frame) "
                   "next to each episode's CSV. Open it with `rerun trajectory.rrd`.")
def main(input_dir, docker_image, output_csv, save_rrd):
    """Run offline OAK-D VSLAM on episode directories.

    Each directory must contain an oak/ subdirectory with:
    calib_offline.json, timestamps.csv, imu_acc.csv, imu_gyro.csv,
    frames/*.png, depth/*.png

    Build the Docker image once with:
        docker build -t pollenrobotics/oak-vslam docker/oak_vslam/

    Produces <episode_dir>/camera_trajectory.csv (absolute poses) compatible
    with generate_dataset.py, plus (unless --no-save-rrd) a trajectory.rrd
    Rerun recording of the 3D trajectory and camera frame.
    """
    ok = 0
    failed = 0
    for d in input_dir:
        ep_dir = Path(d).expanduser().absolute()
        try:
            result = run_oak_slam(
                ep_dir,
                docker_image=docker_image,
                output_csv=output_csv,
                show_progress=True,
            )
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            failed += 1
            continue

        if result.trajectory_path is None:
            print(f"  FAILED: {ep_dir.name} (rc={result.returncode})")
            failed += 1
        else:
            print(f"  -> {result.trajectory_path}")
            ok += 1
            if save_rrd:
                # 3D trajectory + animated camera frame, dropped next to the CSV
                # for later inspection (`rerun <episode>/trajectory.rrd`).
                try:
                    from grabette_postprocess.visualize import save_episode_rrd

                    rrd = save_episode_rrd(ep_dir, traj_csv_name=output_csv)
                    if rrd is not None:
                        print(f"  -> {rrd}")
                except ImportError:
                    print("  (skipped trajectory.rrd: `pip install rerun-sdk`)")
                except Exception as e:
                    print(f"  (could not write trajectory.rrd: {e})")

    if len(input_dir) > 1:
        print(f"\nDone: {ok} OK, {failed} failed")


if __name__ == "__main__":
    main()
