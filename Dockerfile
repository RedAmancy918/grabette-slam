# grabette-slam → LeRobot Space image, built on pollen's pre-built SLAM base
# (pollenrobotics/oak-vslam) which already ships the compiled offline_vslam
# binary + RTAB-Map libraries. This replaces the previous from-source RTAB-Map
# compile (~10 min on every Space build) with a pull of the base image — we add
# only the Python 3.12 conversion app on top. Skipping the compile also drops the
# build-time dependency on the rtabmap source and the heavy -dev packages.
FROM pollenrobotics/oak-vslam

USER root
# The base provides offline_vslam here; the app calls it as a subprocess.
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

# grabette-postprocess package (pulls lerobot). offline_vslam is already compiled
# into the base image, so we only install the Python package here.
COPY grabette-postprocess /opt/grabette-postprocess
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
# The app's modules, listed rather than copied wholesale so the build context's
# other files (the vendored package, docs) never end up in the runtime image.
# ADD A NEW MODULE HERE when you create one: it is not in the image otherwise,
# and the Space then dies on import at startup with ModuleNotFoundError.
COPY --chown=user app.py pipeline.py views.py controller.py review.py session.py \
     gather.py /home/user/app/

EXPOSE 7860
# Override the base image's offline_vslam entrypoint — we run the conversion app,
# which invokes offline_vslam as a subprocess via OAK_VSLAM_BINARY.
ENTRYPOINT []
CMD ["python", "app.py"]
