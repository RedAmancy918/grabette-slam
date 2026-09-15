# grabette-slam → LeRobot Space image, built on pollen's pre-built SLAM base
# (pollenrobotics/oak-vslam), which supplies the RTAB-Map libraries and the C++
# toolchain. This keeps the ~10 min from-source RTAB-Map compile out of every
# Space build — we add the Python 3.12 conversion app, and recompile
# offline_vslam, on top.
#
# offline_vslam is recompiled here rather than taken from the base image. The
# base bundles a binary built from whatever offline_vslam.cpp looked like when
# someone last pushed the image by hand, and nothing rebuilds it when the .cpp
# changes — so the two drift apart silently. They did: the IMU-less calib fix
# was in this tree but not in the base's binary, and every Gemini episode
# aborted (SIGABRT, surfacing as rc=-6) in a Space whose source had the fix.
# Compiling from the deployed tree costs ~10 s and makes the binary match the
# source by construction.
FROM pollenrobotics/oak-vslam

USER root
# offline_vslam is compiled to this path below; the app calls it as a subprocess.
ENV OAK_VSLAM_BINARY=/usr/local/bin/offline_vslam

# Python 3.12 (lerobot requires it; the base ships 3.10) via deadsnakes.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gnupg dirmngr \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# grabette-postprocess package (pulls lerobot), and the offline_vslam source it
# carries.
COPY grabette-postprocess /opt/grabette-postprocess

# Rebuild offline_vslam from the tree above, overwriting the base image's binary.
# Same flags as grabette-postprocess/docker/oak_vslam/Dockerfile, which builds
# the base: the -dev packages, RTAB-Map headers and libs they need are all still
# present there. Before the slow pip layer, so a compile error surfaces at once.
RUN RTABMAP_INCLUDE=$(find /usr/local/include -maxdepth 1 -name "rtabmap*" -type d | head -1) \
 && PCL_INCLUDE=$(find /usr/include -maxdepth 1 -name "pcl*" -type d | head -1) \
 && g++ -std=c++17 -O2 /opt/grabette-postprocess/docker/oak_vslam/offline_vslam.cpp \
        -o /usr/local/bin/offline_vslam \
        -I${RTABMAP_INCLUDE} \
        -I${PCL_INCLUDE} \
        $(pkg-config --cflags eigen3) \
        -lrtabmap_core -lrtabmap_utilite \
        $(pkg-config --cflags --libs opencv4) \
 && chmod +x /usr/local/bin/offline_vslam

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir /opt/grabette-postprocess \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Runtime libs the SLAM base lacks: ffmpeg (lerobot video encode/decode) +
# libgl1/libglib2.0-0 (opencv-python). Placed after the pip layer so the slow
# lerobot install stays cached across rebuilds.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Run as the non-root user Spaces expect (uid 1000), with writable caches.
RUN useradd -m -u 1000 user || true
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    SYSTEM=spaces
# SYSTEM=spaces lets Gradio's get_space() recognize this Docker Space so it uses
# the real HF OAuth flow (vars injected by hf_oauth: true) instead of the local mock.
WORKDIR /home/user/app
COPY --chown=user app.py pipeline.py views.py controller.py review.py session.py /home/user/app/

EXPOSE 7860
# Override the base image's offline_vslam entrypoint — we run the conversion app,
# which invokes offline_vslam as a subprocess via OAK_VSLAM_BINARY.
ENTRYPOINT []
CMD ["python", "app.py"]
