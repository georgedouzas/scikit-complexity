# Tasks

## 1. Inventory and dependency

- [ ] 1.1 Move the inventory entry of `elman_network` from out of scope to `pending add-learning-dynamics`, add Elman
  1990 to its sources, and verify the inventory check passes with the out-of-scope entry moved scenario
- [ ] 1.2 Declare the `learning` extra with `jax>=0.11.2`, `optax>=0.2.8`, and `diffrax>=0.7.2`, teach deptry the
  optional imports, add `MissingExtraError` to the errors module unless it exists, and verify `pdm install -G learning`
  and the checks session pass
- [ ] 1.3 Add a CI job without the extra, and verify it runs the suite with learning tests skipped by a marker

## 2. Matrix-valued parameters

- [ ] 2.1 Add the `shape` argument of `parameter` with role names, the row-major block in the layout, and shape-aware
  `Trajectory` reads and `update_parameters`, and verify the wrong shape scenario
- [ ] 2.2 Let reduction weights read a matrix parameter on the NumPy and Numba backends, and verify the weighted input
  scenario on both

## 3. JAX lowering

- [ ] 3.1 Add `JAX_DIALECT` and `JaxBackend` for discrete maps and ODEs with reductions and matrix parameters, raising
  `BackendError` for opaque callables and `MissingExtraError` without the extra, and verify the weighted input
  scenario on JAX, forward agreement with NumPy within 1e-10 on `logistic_map` and `lotka_volterra`, and the missing
  extra scenario

## 4. Learning systems

- [ ] 4.1 Add `Form.LEARNING`, `LearningSystem`, `Dataset`, and `Optimizer`, and compile a learning system with the
  checks of trained paths, loss, metrics, and dataset shapes, and verify the untrained parameter path, unknown loss,
  shape mismatch, unknown optimizer, and not differentiable scenarios
- [ ] 4.2 Add the `GradientDescent` stepper registered as `'optax'` with `lax.scan` unrolls, diffrax ODE unrolls,
  truncation, double precision, injected hyperparameters, and optimizer state reset, and verify the gradient against
  finite differences, gradient through an ODE, truncation, zero learning rate, and learning rate schedule scenarios
- [ ] 4.3 Run learning systems through `Simulation` with whole-epoch `dt`, seeded shuffling, and the loss and metric
  observables, and verify the epochs as frames, time between frames, loss at the start, masked steps, same seed, and
  updates per epoch scenarios
- [ ] 4.4 Add `build_model`, and verify the trained model agrees with training scenario
- [ ] 4.5 Verify the Lotka-Volterra fit scenario and the Oja's rule scenario, the latter in the job without the extra

## 5. Visualization

- [ ] 5.1 Add `MatrixVisual` registered as `'matrix'`, the learning system's visual, and its primary view and default
  blueprint, and verify the recurrent weights, primary view, and default blueprint scenarios

## 6. Library model

- [ ] 6.1 Add `elman_network` to `computer_science/neural_networks` as a registered factory built only from public core
  names, mark its inventory entry implemented, and verify the temporal XOR and Elman's original rule scenarios, the
  conformance suite, and the inventory check
- [ ] 6.2 Add `docs/examples/computer_science/neural_networks/plot_elman_network.py` with the loss thumbnail and the
  error-by-position figure, and verify the gallery example scenario and the gallery checks

## 7. Documentation and gate

- [ ] 7.1 Document learning systems, datasets, losses, truncation, optimizers, matrix parameters, the extra, and the
  zero gradient of step functions in the user guide, add a changelog entry, add the `learning` extra and the JAX
  printer to the constitution's Project Profile, and verify the documentation build runs the examples
- [ ] 7.2 Run formatting, checks, the docs build, and tests through nox with and without the extra, and verify all of
  them pass
