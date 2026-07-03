ARG AGENT_BASE_IMAGE=vllm/vllm-openai:latest
FROM ${AGENT_BASE_IMAGE}

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/apps/agent/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-fmt-all \
 && rm -rf /var/lib/apt/lists/*

# Voice runtime dependencies are installed before the source tree so code
# edits do not invalidate the multi-gigabyte dependency layer.
COPY apps/agent/constraints.txt /app/apps/agent/constraints.txt
RUN python3 -m pip install --no-cache-dir -c /app/apps/agent/constraints.txt \
    qwen-tts \
    onnxruntime-gpu \
    livekit \
    livekit-api \
    soundfile \
    omegaconf \
    hydra-core \
    pyannote-audio==4.0.0 \
    sentencepiece \
    silero-vad \
    torchcodec==0.7.0 \
    prometheus-client \
 && python3 -m pip uninstall -y onnxruntime onnxruntime-gpu \
 && python3 -m pip install --no-cache-dir onnxruntime-gpu
# ^ qwen-tts drags in CPU onnxruntime which shadows the GPU build; reinstall
#   onnxruntime-gpu last so CUDAExecutionProvider stays available.

COPY apps/agent/pyproject.toml /app/apps/agent/pyproject.toml
COPY apps/agent/src /app/apps/agent/src
COPY apps/web/public /app/apps/web/public
COPY models/manifests /app/models/manifests
COPY packages /app/packages
COPY profiles /app/profiles
COPY scripts /app/scripts

RUN python3 -m pip install --no-cache-dir --no-deps -e /app/apps/agent

EXPOSE 8080

ENTRYPOINT []
CMD ["python3", "-m", "ru_local_avatar_agent"]
