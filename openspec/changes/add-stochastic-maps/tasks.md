# Tasks

## 1. Random draws

- [ ] 1.1 Add `uniform`, `normal`, `bernoulli`, and `categorical` as named Sage functions in `skcomplex.core.model`
  with runnable docstring examples, and verify the doctests pass
- [ ] 1.2 Collect and check the draws of each component in the compiler, and reject draws outside update laws, and
  verify the conflicting draw and draw in a rate law scenarios
- [ ] 1.3 Lower draws to an ordered draw block on the NumPy backend, add `Problem.stochastic` and the
  `update(t, x, p, rng)` signature, and pass the generator from the `iteration` stepper, and verify the uniform draw
  statistics, shared draw, independent members, and state-dependent arguments scenarios
- [ ] 1.4 Lower the draw block to the Numba `iterate` kernel with a `Generator` argument, and verify the backends agree,
  law order, same seed, and compiled speed scenarios, and that deterministic maps keep their kernels and results
- [ ] 1.5 Document draws and their sharing rule in the user guide, and verify the documentation build runs its examples

## 2. Markov chains

- [ ] 2.1 Add `Transition` and `Emission` and their rewriting into categorical updates with constant or parameter
  matrices, and verify the stationary distribution, aligned emission, and emission statistics scenarios against
  `quantecon` as a test dependency
- [ ] 2.2 Add the matrix check at compile time and on live parameter changes, and verify the row that does not sum to
  one, live change breaks a row, and live matrix change scenarios
- [ ] 2.3 Document Markov chains and hidden Markov sources in the user guide, and verify the documentation build runs
  its examples

## 3. Library models

- [ ] 3.1 Add the persistent random walk under `mathematics/stochastic_processes`, and verify its diffusion scenario and
  the conformance suite
- [ ] 3.2 Add the Markov and hidden Markov sources under `mathematics/stochastic_processes`, and verify the entropy
  rate and frequency scenarios and the conformance suite
- [ ] 3.3 Add the chaos game with the Sierpiński and Barnsley presets under `mathematics/fractals`, and verify the
  box-counting scenario, its `PointsVisual` stream in a Rerun recording, and the conformance suite
- [ ] 3.4 Mark the three inventory entries as implemented, add the models to their field pages, and verify the
  inventory check and the layout test pass

## 4. Integration and gate

- [ ] 4.1 Add a changelog entry for the draws, the Markov laws, and the new models, and verify the changelog session
  accepts it
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
