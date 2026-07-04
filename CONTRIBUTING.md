# Contributing To Rula

Rula is a local-first realtime voice-avatar runtime. Contributions are welcome when they keep the project secure, reproducible, and honest about demo vs production readiness.

## Before You Open A PR

- Keep secrets, model weights, generated media, logs, SQLite databases, and local operator notes out of Git.
- Do not add cloud inference to the default runtime path.
- Preserve the realtime hot-path envelope: `session_id`, `turn_id`, `generation_id`, `branch_state`, `seq`, and optional `pts_ms`.
- Drop stale `generation_id` artifacts silently at every hop.
- Keep the runtime a modular monolith unless there is measured scale pressure.
- Prefer small typed contracts over framework magic.

## Development Checks

Run the focused checks before submitting changes:

```powershell
$env:PYTHONPATH=(Resolve-Path ".\apps\agent\src").Path
python -m unittest discover apps\agent\tests
npm run protocol:test
npm run web:test
npm run web:build
```

For local Docker validation:

```bash
cd infra/wsl
docker compose config --quiet
```

## Pull Request Expectations

- Explain the user-visible behavior or operational risk being changed.
- Include tests for protocol, turn-state, interruption, API, or UI behavior when touched.
- Keep unrelated refactors out of the PR.
- Call out model, license, GPU, latency, or air-gap implications explicitly.
- Do not paste tokens, private paths, hostnames, SSH config, or local deployment runbooks.

## Good First Areas

- README and release evidence improvements.
- Local eval scripts and acceptance reporting.
- Protocol tests around stale-generation dropping.
- Browser-side audio/face sync diagnostics.
- Documentation for model licensing and attribution.

