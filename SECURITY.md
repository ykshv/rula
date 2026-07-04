# Security Policy

Rula is an early demo runtime, but the repository should still be treated as public production-facing code.

## Supported Scope

Security reports are in scope for:

- FastAPI endpoints and admin routes.
- Local configuration and secret handling.
- Browser client security issues.
- Dependency and supply-chain risks.
- Unsafe file handling, path traversal, SSRF, XSS, injection, auth bypass, or secret leakage.
- Realtime protocol issues that can replay stale audio, face frames, or session state across generations.

Model behavior, model licensing, and generated content safety are tracked separately through the legal and eval gates, unless the issue causes a concrete software vulnerability.

## Reporting

Please open a private security advisory on GitHub when possible.

If a public issue is the only available channel, do not include:

- tokens or credentials;
- private hostnames or IPs;
- SSH keys;
- local deployment paths;
- conversation logs containing personal data;
- exploit details that make active abuse easier.

Include the affected commit, reproduction steps, expected impact, and the smallest safe proof of concept.

## Defaults

- The public repo must not contain `.env.local`, `infra/wsl/.env`, model weights, generated media, SQLite databases, logs, certificates, SSH keys, or local operator notes.
- Production mode should fail closed around readiness and admin access.
- Public demo exposure must not be treated as the target closed-contour deployment model.

