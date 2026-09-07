---
title: Grabette Slam
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_scopes:
  - read-repos
  - write-repos
  - manage-repos
---

# Grabette SLAM → LeRobot

A HuggingFace Space that turns a raw Grabette recording dataset into a
[LeRobot](https://github.com/huggingface/lerobot) dataset and pushes it to the Hub.

1. **Sign in** with your Hugging Face account (OAuth).
2. Give a **source** dataset `repo_id` (raw OAK-D recording on the Hub), a
   **target** `repo_id` to create, and a task description.
3. For each episode the Space runs, in-process:
   - `convert_episode` — expand the recording into the `oak/` layout
   - `run_oak_slam` — RGBD-inertial odometry via the bundled `offline_vslam` binary
   - `build_dataset` — assemble a LeRobot v3 dataset
   - `push_to_hub` — upload under your account
4. A link + embedded view of the
   [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset)
   is shown for the result (the dataset must be public to be visualized).

## Community contributions (folded panel, above the form)

A **community dataset** holds no recordings: each accepted episode is one small
JSON under `contrib/<contributor>/` pointing at a recording that lives in the
contributor's own raw dataset. See
[the fleet's docs](https://huggingface.co/spaces/pollen-robotics/grabette-fleet/blob/main/docs/community-datasets.md)
for why it is shaped that way.

Building a LeRobot dataset from one is therefore two steps, and this Space does
both — separately, or on one click.

1. **Gather** — give it the community dataset and a raw dataset of your own. It
   reads every merged manifest, fetches the episodes from each contributor's repo
   and copies them in under `{episode_id}__{contributor}/{role}/`. It produces a
   **raw** dataset, not a LeRobot one. Re-runnable: an episode already gathered is
   skipped, so merging one more pull request means gathering again, not
   rebuilding. Tick **Dry run** to see the plan first, and set **Required
   devices** so a contribution short of an arm is left out and named rather than
   half-copied.
2. **Convert** — the usual run. The gather fills in **Source dataset** for you,
   so it is one click away.

Ticking **…and convert to a LeRobot dataset straight after** does both from the
gather button: the conversion starts on the raw dataset just written, with the
Target dataset and Task description from the form below. Those two are validated
*before* any copying, since discovering an empty target after fetching every
episode is the worst possible ordering. A dry run never chains, and neither does
a failed gather.

It runs here rather than on a laptop for a plain reason: both ends are on the
Hub, so gathering from home would pull every episode down a domestic link and
push it straight back up. The same logic is available as a CLI
(`scripts/pipeline/collect_community_contributions.py` in `grabette-postprocess`)
for a one-off from a machine with a fat pipe.

## Why a Docker Space

The SLAM step is a compiled C++ binary (`offline_vslam`, built on RTAB-Map).
Locally it runs in the `pollenrobotics/oak-vslam` Docker image, but Spaces cannot
run Docker-in-Docker. So this Space **is** that image: it builds the binary at
image-build time and `run_oak_slam(..., binary=...)` calls it directly.

## Self-contained build

The `grabette-postprocess` package (with `offline_vslam.cpp`) is vendored into
the build context as `./grabette-postprocess` — no git clone, no private-repo
secret. `deploy.sh` assembles that layout from the working tree and uploads it.

```bash
HF="uv run --project ../../packages/grabette-postprocess hf" \
  ./deploy.sh pollen-robotics/grabette-slam
```

> Note: RTAB-Map is compiled with `--parallel 2` to stay within the HF build
> runner's memory (full parallelism OOMs it).
