## Summary

Describe the behavior, architecture, or documentation change.

## Verification

- [ ] Python unit tests
- [ ] Protocol tests
- [ ] Web typecheck/build
- [ ] Docker compose config
- [ ] Secret/path scan
- [ ] Not applicable, with reason:

## Runtime Impact

- [ ] Hot-path envelope preserved: `session_id`, `turn_id`, `generation_id`, `branch_state`, `seq`, `pts_ms?`
- [ ] Stale-generation dropping preserved
- [ ] No cloud inference added to the default runtime
- [ ] No secrets, model weights, generated media, local paths, or operator runbooks added
- [ ] License / model / asset implications considered

