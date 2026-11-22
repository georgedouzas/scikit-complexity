## Why

TorchPhysics (Qewton Labs, formerly Bosch Research, Apache-2.0) trains neural networks on the equations of a field
instead of on its runs. Two of its methods have no counterpart in the planned stack. The Deep Ritz method finds a
static field by minimising its energy, the form an `ObjectiveLaw` over a field already states. Its parameter spaces
train one network for a whole range of parameters, so a sweep over a field system costs one training instead of one
solve per point. `add-torch-surrogates` already covers single physics-informed solves with deepxde, so TorchPhysics is
added for these two methods only, and physics-informed networks keep one implementation.

**Postponed.** TorchPhysics 1.1.2, its latest release (2026-06-12), pins `numpy<2.0` and requires `jupyter` and
`pytorch-lightning` at install. The package needs `numpy>=2.1` on Python 3.12 to 3.14, and numpy 1.x has no wheels
after Python 3.12, so the two cannot be installed together. This change starts only when a TorchPhysics release
allows numpy 2. Until then it stays a recorded plan.

## What Changes

- **Deep Ritz solver.** New solver `DeepRitzSolver` finds the static state of a field system whose laws give its energy
  as `ObjectiveLaw`, such as the Dirichlet energy of a membrane under load, by training a network on the energy
  integral over the domain of the space, with the fixed values of the boundary of the space as a penalty. It is the
  `solver` setting of the system, prepared by `system.prepare()` like every other solver, and `simulate` returns the
  field on the points of the space.
- **Parametric physics-informed surrogates.** New analysis function `parametric_surrogate(system, params=...)` trains
  one physics-informed network over the field and the ranges of the parameters named as `get_params` names them, such
  as `params={'k': (0.5, 2.0)}` for the diffusion coefficient of a ready-made heat system or
  `params={'diffusion__k': (0.5, 2.0)}` for a `System`. Its result follows the surrogate convention `system`,
  `training_error`, `validation_error`. The `system` gives the field at any parameter value of the range without a new
  solve, and the validation error compares it with solves of the system itself at held out parameter values.
- **Dependency.** `torchphysics` becomes a runtime dependency, since the core uses it. It brings `pytorch-lightning`.
  The torch it needs is the one `add-torch-surrogates` already adds.
- **No second way.** Single physics-informed solves of rate laws stay with `PinnSolver` on deepxde. TorchPhysics is not
  used for them.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `pde-solvers`: static fields given by an energy are solved by the Deep Ritz method.
- `data-driven-dynamics`: physics-informed surrogates that cover a range of parameters with one training.

## Impact

- Core: new `src/skcomplex/core/engine/solvers/_deep_ritz.py` and `src/skcomplex/core/analysis/_parametric_surrogate.py`,
  with mirrored tests.
- API: new public `DeepRitzSolver` in `skcomplex.core.engine.solvers` and `parametric_surrogate` in
  `skcomplex.core.analysis`. Nothing existing changes.
- Dependencies: `torchphysics` (1.1.2 or the first release that allows numpy 2) and its `pytorch-lightning`. The
  install grows by tens of megabytes over the torch stack of `add-torch-surrogates`.
- Depends on `add-torch-surrogates` (torch, CPU wheels in CI, `PinnSolver` as the single solve path). It builds on
  `System`, `prepare`, `simulate`, and `ObjectiveLaw`, already in the code since `redesign-core-api` was archived.
- Blocked on upstream: a TorchPhysics release without the `numpy<2.0` pin. Its hard `jupyter` requirement should go in
  the same release, or the change records why the package accepts it.
- Gallery: one example for each method, a loaded membrane and a diffusion sweep.
