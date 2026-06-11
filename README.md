---
title: Grabette SLAM Pipeline
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Grabette SLAM (test mode)

A HuggingFace Space that runs the OAK-D SLAM pipeline on a raw recording dataset
and visualizes the trajectories. Current scope stops at SLAM:

1. **Upload** a `.zip` of the dataset
2. `convert_episode` — expand each episode into the `oak/` layout
3. `run_oak_slam` — RGBD-inertial odometry via the bundled `offline_vslam` binary
4. **Visualize** each `camera_trajectory.csv` as a gravity-aligned 3D plot

The LeRobot build + Hub push are intentionally disabled for now
(`process_dataset` in `pipeline.py` keeps that path for later — see the
`angle_data.json` caveat there before re-enabling it).

## Why a Docker Space

The SLAM step is a compiled C++ binary (`offline_vslam`, built on RTAB-Map).
Locally it runs in the `pollenrobotics/oak-vslam` Docker image, but Spaces cannot
run Docker-in-Docker. So this Space **is** that image: it builds the binary at
image-build time and `run_oak_slam(..., binary=...)` calls it directly.

## Build configuration

The image clones the grabette repo to get `offline_vslam.cpp` and the
`grabette-postprocess` package. Pin the ref via build args:

- `GRABETTE_REPO` (default `https://github.com/pollen-robotics/grabette.git`)
- `GRABETTE_REF` (default `main`)

If the repo is private, add a build-time secret token and inject it into the
`git clone` URL (see the HF Spaces "Build secrets" docs).

## Run locally

```bash
docker build -t grabette-slam \
  --build-arg GRABETTE_REF=<your-branch> spaces/grabette-slam
docker run --rm -p 7860:7860 grabette-slam
# open http://localhost:7860, upload a dataset .zip
```
