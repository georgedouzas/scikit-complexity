## Why

`BaseSpec` derives from `BaseConfig`, so every frozen value carries `prepare` and `run` that do nothing and return the
spec unchanged. In return every part, solver, and compiler carries a `_check` hook beside `_prepare`, and a `_validated`
wrapper, only because specs and estimators share one base. Two kinds of object are forced through one interface. A
spec is a value that is validated by its owner. An estimator is configured, prepared against a context, and run.

## What Changes

- **BREAKING: specs are not estimators.** `BaseSpec` no longer derives from `BaseConfig`. It has no `prepare` and no
  `run`, only the private hook `_check`, which its owner calls when it reads the value. `spec.prepare()` and
  `spec.run()` are removed.
- **Estimators validate in `_prepare`.** `BaseConfig` keeps `prepare` and `run`, with the hooks `_prepare` and `_run`.
  `_check` and `_validated` are removed from it. Every estimator checks its own settings at the start of its
  `_prepare`, before it derives anything, and a subclass chains with `super()._prepare(context)` as it chains
  `super()._check()` today.
- **One root for everything that acts.** Parts, solvers, backends, compilers, passes, observers, visuals, clocks,
  printers, kernels, and `Calibration` stay on `BaseConfig`. What each family adds stays on its own base: `BasePart`
  adds tags, laws, roles, and nesting, `BaseSolver` and `BaseBackend` add supported tags. No third root is added.
- **Owners prepare what they own.** An owner that validated a part without preparing it now prepares it, or reads the
  specs it holds, whose `_check` it calls.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`: two bases, `BaseConfig` for estimators with `prepare` and `run` and `BaseSpec` for frozen values
  with `_check` only, and validation at the start of `_prepare`.

## Impact

Every spec class (56 `_check` definitions among them) loses `prepare` and `run`, and every caller of them changes.
About 160 estimators move their `_check` into `_prepare`. Breaking for anyone who prepared or ran a spec or wrote a
custom part with `_check`, which the migration page of 0.7 covers. Behaviour of systems, solvers, and analyses does not
change.
