# Tasks

## 1. Dependencies

- [x] 1.1 Add `jax>=0.11.2`, `optax>=0.2.8`, and `diffrax>=0.7.2` to the runtime dependencies in `pyproject.toml`, and
  name them, the JAX backend, and the macOS platform on Apple silicon in the Project Profile; verify `pdm install` and
  the checks session, including `deptry`, pass

## 2. Shaped parameters

- [x] 2.1 Add the `shape` field of `ParameterSpec` with role names, the row-major block in the parameter layout,
  shape-aware `Trajectory.values` and `Simulation.update_parameters`, and the `parameter-shape` issue; verify the wrong
  shape and unbound roles scenarios
- [x] 2.2 Let the weights of `total` and `mean` read a shaped parameter on the NumPy and Numba backends; verify the
  weighted input scenario on both
- [x] 2.3 Let `CouplingInteraction(mode='group')` take a shaped parameter as `weights` with a function of both members,
  lowered to a matrix product when the function factors and to a pair loop otherwise; verify the generalized
  Lotka-Volterra coupling scenario on NumPy and Numba

- [x] 2.4 Add the reduction `contraction` to `skcomplex.core.laws.utils` with its NumPy and Numba lowering, and the
  `parameter-shape` issue of a tensor on an element; verify the contraction against NumPy and tensor on an element
  scenarios on NumPy and Numba

## 3. JAX backend

- [x] 3.1 Add `JaxBackend` in `engine/backends/_jax.py` with its `DialectSpec`, for discrete maps and ODEs with
  reductions and shaped parameters, and its `SupportSpec`; verify the weighted input and coupling scenarios on JAX and
  forward agreement with NumPy within 1e-10 on `logistic_map` and `lotka_volterra`, and the contraction against NumPy
  scenario on JAX

## 4. Training law and learning solver

- [x] 4.1 Add `GradientStepLaw` in `core/laws/_gradient_step.py`, `TrainingInteraction` in
  `core/interactions/_training.py` with `OPTIMIZERS` and its hyperparameters, `DatasetSpec` in `skcomplex.specs`, and
  `FormKind.LEARNING`; verify the unknown optimizer scenario and that `GradientInteraction` is unchanged
- [x] 4.2 Compile a system holding the law into a `LearningProblemSpec` with the nested system as its inner problem,
  and add the learning issues to the pass compiler; verify the untrained parameter path, parameter not adaptive,
  unknown loss, shape mismatch, and not differentiable scenarios
- [x] 4.3 Add `GradientDescentSolver` in `engine/solvers/_gradient_descent.py` with `lax.scan` unrolls, diffrax ODE
  unrolls, truncation, double precision, injected hyperparameters, and optimizer state reset, and map
  `FormKind.LEARNING` to it in `DEFAULT_SOLVERS`; verify the gradient against finite differences, gradient through an
  ODE, truncation, zero learning rate, learning rate schedule, automatic solver, and unsupported solver scenarios
- [x] 4.4 Run systems holding the law through `Simulation` with whole-epoch `dt`, seeded shuffling, and the recorded
  loss and metrics; verify the epochs as frames, time between frames, loss at the start, masked steps, same seed, and
  updates per epoch scenarios
- [x] 4.5 Verify the trained model agrees with training, Lotka-Volterra fit, Oja's rule, and tensor trains scenarios

## 5. Visualization

- [x] 5.1 Add `MatrixVisual` in `engine/viz/_matrices.py`, chosen by `VisualSpec('matrix')`, the visual of a system
  that trains, and its place first in the default blueprint of `RerunObserver`; verify the recurrent weights, first
  view, and default blueprint scenarios

## 6. Models

- [x] 6.1 Add `elman_network` to `computer_science/neural_networks` built only from public names, with its catalog
  observables, and mark its inventory entry implemented; verify the temporal XOR and Elman's original rule scenarios
  and the inventory check
- [ ] 6.2 Add `docs/examples/computer_science/neural_networks/plot_elman_network.py` with the loss thumbnail and the
  error-by-position figure; verify the gallery example scenario and the gallery checks
- [x] 6.3 Rebuild `GeneralizedLotkaVolterraInteraction` on `CouplingInteraction(mode='group')` with a shaped matrix
  parameter and the growth as a `RateInteraction`; verify the rebuilt community and matrix swept scenarios
- [ ] 6.4 Add `higher_order_random_lotka_volterra` to `biology/ecology` on a shaped tensor parameter and `contraction`,
  with a curated gallery example, and mark its inventory entry implemented in place of `pending
  add-higher-order-interactions`, with the tags of the system it builds, a group with no network and no hyperedge
  binding; verify the no interactions and random community scenarios, the conformance suite, and the inventory
  check

## 7. Documentation and gate

- [ ] 7.1 Document the training interaction and its law, its difference from `GradientInteraction`, nested inner
  models, datasets, losses, truncation, optimizers, shaped parameters, and the zero gradient of step functions in the
  user guide, and add the changelog entry; verify the documentation build runs the examples
- [ ] 7.2 Run formatting, checks, the docs build, and tests through nox; verify all of them pass
