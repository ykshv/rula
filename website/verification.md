---
title: Verification
description: Rula demo status, latency targets, GPU and air-gap acceptance gates, and the distinction between a public website and a working voice runtime.
---
# Verification

Rula is a demo / early runtime. The limits below are acceptance targets, not measurements established by the documentation build.

As checked on 2026-10-25, `ai.igor-ya.ru` serves a demo-unavailable page with `noindex`. The canonical project website is [this documentation](index.md); a running public voice session is not claimed.

## API And Evals
Core endpoints: `GET /health`, `GET /ready`, `GET /api/runtime/status`, `GET /metrics`, `POST /api/sessions`, `POST /api/chat/text`, admin cancellation and conversation/turn trace endpoints, and `GET /api/acceptance`.

Local evidence path:

- `/metrics` exposes Prometheus-compatible latency and reliability metrics.
- `TurnTrace` records EOT, speculation, first LLM token, first TTS chunk, first audio, and audio completion.
- `SQLiteConversationAuditStore` persists local conversation events and state snapshots.
- `scripts/evals/e2e_probe.py` measures first audio, barge-in, speculative hit rate, and data-channel envelopes.
- `voice_smoke.py`, `gpu_smoke.py`, `latency_report.py`, `soak_test.py`, and `legal_gate.py` define release evidence.

## Quality Targets
| Gate | Target |
|---|---|
| First audio p50 | 600-700 ms after user speech end |
| First audio p95 | <= 1100 ms |
| Avatar visible reaction | <= 250 ms |
| Barge-in p95 | <= 300 ms |
| Russian ASR WER | <= 8% |
| Speculative hit rate | >= 70% |
| Audio-face PTS drift | <= 50 ms p95 |
| Reliability | 60-minute or 120+ turn soak test |

Run acceptance checks before calling any deployment production-ready:

```powershell
python .\scripts\evals\legal_gate.py
python .\scripts\evals\gpu_smoke.py
python .\scripts\evals\latency_report.py
python .\scripts\evals\soak_test.py
powershell -ExecutionPolicy Bypass -File .\scripts\airgap\check_windows_firewall.ps1
```

From WSL:

```bash
bash scripts/airgap/check_wsl_network.sh
```

## What Is Still Missing
- Current legal evidence for every model, asset, container image, and redistribution path.
- Full 60-minute / 120+ turn soak evidence on the target machine.
- Verified air-gap run where the full conversation works with outbound network blocked.
- Release manifest for model and container checksums.
- Stronger browser/client security headers if the demo UI is exposed outside localhost.
- Photorealistic or video-avatar visual layer if the target is a realistic human clone.
- Tool-calling layer with explicit schemas, permission policy, timeouts, and audit trail.
- Durable long-term memory or local RAG with privacy boundaries and evals.
- Offline dependency mirror and reproducible closed-contour install bundle.
- Operational runbooks for GPU OOM, LiveKit failures, vLLM failures, TTS stalls, and corrupted model cache.
- Third-party attribution bundle for redistributed assets and model artifacts.

## Licensing and privacy

The MIT license covers repository code and documentation. Model weights, voice likenesses, avatar assets and external runtimes retain their own terms. Keep tokens, model artifacts, generated media and conversation data out of public reports.
