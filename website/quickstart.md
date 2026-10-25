---
title: Get started
description: Run the Rula voice-avatar demo locally on a Windows, WSL2 and NVIDIA GPU workstation; download models, start services and check readiness.
---
# Get started

The public documentation needs no GPU or voice backend. These steps start the separate local runtime.

## Tech Stack
- **Backend:** Python 3.11, FastAPI, Pydantic, Uvicorn.
- **Voice runtime:** in-process STT/TTS/A2F workers, LiveKit Python SDK, ONNX Runtime GPU.
- **LLM serving:** local vLLM server.
- **Frontend:** React, TypeScript, Vite, Three.js, `@pixiv/three-vrm`, LiveKit client.
- **Host profile:** Windows checkout, WSL2/Docker GPU runtime, NVIDIA CUDA.
- **Observability:** Prometheus and Grafana in the WSL compose stack.

## Quick Start
Recommended target: Windows + WSL2, NVIDIA CUDA GPU, Docker Desktop with GPU support, 64 GB RAM minimum, 128 GB recommended, and a Hugging Face token.

```powershell
git clone https://github.com/ykshv/rula.git
cd rula
powershell -ExecutionPolicy Bypass -File .\scripts\secrets\set_hf_token.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\models\download_all.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\assets\download_default_avatar.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\dev\start_local.ps1
```

Open `http://127.0.0.1:46174/`.

Stop:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev\stop_local.ps1
```

From WSL:

```bash
cd /mnt/<drive>/path/to/rula/infra/wsl
cp .env.example .env
docker compose up --build
```

Default local endpoints: Web UI `:46174`, Agent API `:46181`, vLLM `:46111`, LiveKit `:46280`, Prometheus `:46909`, Grafana `:46300`.
