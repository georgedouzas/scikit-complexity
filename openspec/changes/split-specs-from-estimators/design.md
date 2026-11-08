## Context

`BaseConfig` (src/skcomplex/core/base.py) defines `prepare`, which calls `_validated()` (that is `_check()`) and then
`_prepare(context)`, and `run`, which calls `_run`. `BaseSpec` derives from it and implements the three hooks as no-ops,
so a spec has two public methods that do nothing. Owners call `_validated()` on parts and specs they do not prepare.

## Goals / Non-Goals

**Goals:**
- Specs are values: fields, readers of their fields, and `_check`. Nothing else.
- Estimators have one validation place, the start of `_prepare`.
- The hierarchy says what each kind is: `BaseConfig` for what acts, `BaseSpec` for what is.

**Non-Goals:**
- A third root for solvers, backends, and compilers. They have a real `prepare(context)` and `run(inputs)` and share
  `BaseConfig` with parts.
- Changing any behaviour of systems, solvers, or analyses.

## Decisions

- **Hierarchy.**
  ```
  BaseConfig        get_params, set_params, clone, prepare, run, hooks _prepare and _run
  ├── BasePart      tags, laws, roles, nesting       → elements, interactions, laws, spaces, times, systems
  ├── BaseSolver, BaseBackend, BaseCompiler, BasePass, BaseLowering
  ├── BaseObserver, BaseVisual, BaseClock, BaseKernel, BasePrinter, BaseSearch
  BaseSpec          frozen dataclass, hook _check   → StateSpec, RoleSpec, Tags, ProblemSpec, FrameSpec, …
  ```
- **`BaseSpec`** is a plain frozen dataclass base with `_check`, which its owner calls, through one private helper of
  the owner's base so owners do not reach into private methods of other classes more than they do today.
  `get_params` and `clone` are not needed on values, which are replaced with `dataclasses.replace`.
- **Validation in `_prepare`.** Each estimator's former `_check` body moves to the top of its `_prepare`, before any
  attribute ending in `_` is set. A family base that implemented `_check` for its family (passes, solvers) does it at
  the top of its `_prepare`. `BasePart` keeps validating its label and declared specs, at the top of its `_prepare`.
- **Owners.** `System` prepares its parts in its `_prepare` instead of validating them first. A compiler prepares the
  system. `BaseSystem` builds its composite and prepares it. Places that call `_validated()` on a spec call `_check()`.
- **Partial preparation.** Today validation of every part runs before any part is prepared, so a failure leaves no
  attribute ending in `_` on any part. With validation inside `_prepare`, a system whose third part is invalid has its
  first two parts prepared when the error is raised. The requirement "Preparing validates first" changes to the
  object itself: an estimator raises before it sets any attribute of its own. Re-preparing replaces the attributes, so
  a partly prepared system is harmless.

- **Calibration as a function.** `calibrate` takes the arguments of the former constructor and returns
  `CalibrationResult`. A meta-system that fits lazily, in the role of `GridSearchCV` composed in a pipeline, can be
  added later on top of `calibrate` if a need appears. Today every caller runs it once and reads the fitted system.

## Risks / Trade-offs

- **Size:** about 160 estimator hooks and 56 spec hooks move or change, plus every `spec.prepare()` caller and the
  base tests. It runs after the docstring wave, by field, with the checks per batch.
- **Weaker guarantee on composites,** as described above. Accepted, since nothing reads a part of a system whose
  preparation failed.
