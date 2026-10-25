---
title: Rula — local Russian voice avatar
description: Rula is an open-source local Russian voice-avatar runtime by Igor Yakushev, combining STT, Qwen, TTS, Audio2Face and a LiveKit VRM interface.
---
# Rula

**Local-first realtime Russian voice avatar with a browser VRM interface.**

Speak Russian. Get a local voice response from an animated avatar.

Rula connects Russian speech recognition, a local Qwen language model, local speech synthesis and audio-driven facial animation. LiveKit carries realtime audio and events to a React / Three.js client rendering a 3D VRM avatar.

**Development preview.** The runtime is intended for a Windows / WSL2 / NVIDIA GPU workstation. Its latency, GPU, soak, licensing and air-gap checks must pass on the target host before a production claim. This website documents the project and remains available without an interactive voice server.

## Start here

- [Run the application](quickstart.md): workstation requirements, model downloads and local services.
- [Understand the architecture](architecture.md): turn state, local models, cancellation and media timing.
- [Check verification status](verification.md): targets, evidence requirements and current limitations.
- [Инструкция на русском](ru.md).

## Common questions

### Does Rula require cloud inference?

The default runtime uses local STT, LLM, TTS and face-animation services. Internet access is used to obtain dependencies and model artifacts. A completed air-gap acceptance run is a separate requirement.

### Is the avatar a photorealistic video clone?

The current client renders a 3D VRM avatar driven by audio-based facial animation. A photorealistic video renderer is not part of the current implementation.

### Where is the public demo?

The former interactive endpoint at `ai.igor-ya.ru` currently displays a demo-unavailable page. Use the [local setup](quickstart.md) to run the application. This documentation is the canonical public project website.

## Author and related work

Created and maintained by [Igor Yakushev](https://igor-ya.ru/). Related writing: [On-prem voice agents](https://igor-ya.ru/posts/on-prem-voice-agents/) (Russian).

[Rokid Physical AI](https://ykshv.github.io/rokid-physical-ai/) is another project by the same author: a voice-and-camera interface connecting smart glasses to robot arms.

Source code: [ykshv/rula](https://github.com/ykshv/rula). MIT for repository code and documentation; external models and assets retain their own licenses.
