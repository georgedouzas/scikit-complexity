# Tasks

## 1. Upstream gate

- [ ] 1.1 Open a pull request to Qewton-Labs/torchphysics that allows numpy 2 and moves `jupyter` to an extra, and stop
  the change until a release contains it. Verify that `pip install torchphysics` beside `numpy>=2.1` resolves on
  Python 3.14 from the package index
- [ ] 1.2 Check that `add-torch-surrogates` is applied, so torch and `PinnSolver` exist, and stop if it is not

## 2. Dependency

- [ ] 2.1 Add `torchphysics` at the first release that allows numpy 2 to the runtime dependencies of `pyproject.toml`.
  Verify that `pdm lock` resolves with the current numpy, torch, and Python pins
- [ ] 2.2 Record the dependency with its release date, and any accepted transitive weight such as `jupyter`, in the
  Project Profile of `openspec/constitution.md`. Verify with the dependency audit of the gate
- [ ] 2.3 Make the symbolic to torch conversion of `PinnSolver` a shared private helper if it is not one already.
  Verify with the existing `PinnSolver` tests unchanged

## 3. Deep Ritz solver

- [ ] 3.1 Add `DeepRitzSolver` in `src/skcomplex/core/engine/solvers/_deep_ritz.py`, lowering the space to a domain, the
  energy to an integral condition, and the fixed boundary values to a penalty, and training with Lightning logging and
  checkpoints off. Test the loaded membrane scenario in `tests/core/engine/solvers/test_deep_ritz.py`, marked slow
  where it passes 10 seconds
- [ ] 3.2 Reject at `prepare`, in the support check of the solver, a system whose laws are not one energy over a field.
  Test the heat equation scenario in `tests/core/engine/solvers/test_deep_ritz.py` and check that the message names
  its rate laws
- [ ] 3.3 Compare with finite elements on the same square. Test the scenario of agreement within 2e-2
- [ ] 3.4 Add the docstring with a short example, export the solver, and add the gallery example of the loaded
  membrane. Verify with the doctests and the gallery sample

## 4. Parametric surrogates

- [ ] 4.1 Add `parametric_surrogate` in `src/skcomplex/core/analysis/_parametric_surrogate.py`, lowering the named
  parameters to parameter variables of the network, and return the surrogate result with `system`,
  `training_error`, and `validation_error`. Test the diffusion sweep scenario in
  `tests/core/analysis/test_parametric_surrogate.py`
- [ ] 4.2 Compute the validation error against solves of the system at held out parameter values. Test the validation
  scenario
- [ ] 4.3 Make the surrogate system reject at `prepare` a parameter value outside its range. Test the scenario at
  `k = 3.0` and the message naming `k` and its range in `tests/core/analysis/test_parametric_surrogate.py`
- [ ] 4.4 Add the docstring with a short example, export the function, and add the gallery example of the diffusion
  sweep. Verify with the doctests and the gallery sample

## 5. Gate

- [ ] 5.1 Run the full nox sessions for quality, types, conventions, tests, and the gallery sample. Verify that all pass
