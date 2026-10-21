# Third-Party Notices

GRABETTE SLAM is © Pollen Robotics and released under the Apache License 2.0
(see [`LICENSE`](LICENSE)). It incorporates, depends on, or derives from the
third-party components below, each under its own license. This file is provided
for attribution and license compliance.

---

### RTAB-Map — BSD-3-Clause (Mathieu Labbe, IntRoLab, Université de Sherbrooke)
<https://github.com/introlab/rtabmap>

Built from source (v0.23.1) inside the SLAM Docker image
(`grabette-postprocess/docker/vslam/`); the offline odometry binary
links RTAB-Map's core libraries.

### Luxonis DepthAI — MIT
<https://github.com/luxonis/depthai-core>

`grabette-postprocess/docker/vslam/offline_vslam.cpp` adapts the processing
logic of DepthAI's `RTABMapVIO` node (offline reimplementation; no DepthAI code
is linked at build time).

### LeRobot — Apache-2.0 (Hugging Face, Inc.)
<https://github.com/huggingface/lerobot>

Used as a dependency for LeRobot-format dataset generation.
