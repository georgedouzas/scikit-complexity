# Tasks

## 1. Values and variants

- [x] 1.1 Report a real value outside a real domain as outside it, and keep the imaginary part message for complex values, verified by new tests in `tests/core/engine/compile/test_complex.py` for `-1.0` in `[0, +oo)`, `1.5` in the nonnegative integers, and `1 + 2j` in a real state
- [x] 1.2 Make an unknown variant list only the quoted names and describe accepted specs in words, and update the pinned message in `tests/core/components/interactions/test_base.py`, verified by a test that `MapInteraction(mode='add')` names `'replace'` and `'increment'` and holds no `BaseSpec`

## 2. Laws

- [x] 2.1 Run the order check of rate and noise laws before the early return of the orders pass, verified by a test in `tests/core/components/laws/test_rate.py` that a lone first-order state with `RateLaw(order=2)` raises `ValidationError`
- [x] 2.2 Make `Source` and `Sink` frozen dataclasses with the field `label`, verified by tests in `tests/core/components/laws/test_transfer.py` of their repr, their equality, and an error message that shows `Sink(label='environment')`
- [x] 2.3 Raise `UnregisteredLawError` in the plan pass for a law class with no form, before the mixed laws issue, verified by a test with a law class the engine has no form for, and update its troubleshooting section to say it is raised

## 3. Spaces

- [x] 3.1 Check the points and cells of `MeshSpace` when prepared, verified by tests in `tests/core/components/spaces/test_mesh.py` for one-dimensional points, transposed points, empty cells, one-dimensional cells, and out-of-range indices, each a `ValidationError` naming the argument
- [x] 3.2 Route both layer lookups of `NetworkSpace` through the existing layer check, verified by a test in `tests/core/components/spaces/test_adjacency.py` that a neighbourhood naming a missing layer raises `SpaceError` listing the layers

## 4. Errors

- [ ] 4.1 Remove `ArgumentError`, raise `ValidationError` in the ridesharing system instead, drop its troubleshooting section and its uses in `tests/core/test_errors.py`, verified by those tests, `tests/systems/engineering/traffic`, and `test_troubleshooting_covers_every_error`
- [ ] 4.2 Add a test in `tests/core/test_errors.py` that every public error class of `skcomplex.errors` is raised somewhere in `src`, verified by running it

## 5. Experiments and analyses

- [ ] 5.1 Draw integer starts for paths whose domain holds only integers in the design that basin stability uses, verified by a test in `tests/core/experiments/test_basin_stability.py` on a counting system and by the existing seeded double well tests passing unchanged
- [ ] 5.2 Read the step of `correlation` from the times of a trajectory, with `dt` only for arrays, verified by tests in `tests/core/analysis/test_correlation.py` of a trajectory sampled every 0.1 and of an array with `dt=0.5`, and by the doctests and the markdown blocks of `analyse.md`

## 6. Integration

- [ ] 6.1 Run ruff, mypy on src, tests and docs, lint-imports, the tests in chunks with `-n 4`, the doctests, and the markdown blocks, and verify all pass with no suppression added
