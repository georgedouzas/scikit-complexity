# Tasks

## 1. Groups in the model

- [x] 1.1 Add `Group` and `Element.group`, re-export `Group` from `skcomplex.model`, and verify tests for stored
  arguments and wrong value counts pass
- [x] 1.2 Add the `Boundary` law kind and `PairInteraction` base class, and verify tests for their declarations pass

## 2. Compilation

- [x] 2.1 Extend `Layout` and `Entry` with group shapes, and verify `read` returns `(N,)` and `(N, n_dim)` arrays in
  tests
- [x] 2.2 Lower per-member laws once per group with broadcasting, and verify a test that compiles 10 and 100000
  members within a factor of 2 of each other
- [x] 2.3 Lower pair interactions with `cKDTree` neighbour search and `np.add.at` accumulation, and verify tests for the
  pair force sum and momentum conservation pass
- [x] 2.4 Lower boundary laws to one function excluded from form inference, and verify tests for reflecting and
  periodic boxes pass

## 3. Runtime and visualization

- [x] 3.1 Apply the boundary function after every `advance` and return group arrays from frames and trajectories, and
  verify the trajectory shape and live group parameter tests pass
- [x] 3.2 Log groups as one batch with `color_by`, and verify a test that counts one `Points2D` per frame for a group of
  10000 passes

## 4. Model library and documentation

- [x] 4.1 Add `ReflectingBox`, `PeriodicBox`, and `ideal_gas` to `models.physics`, and verify the ideal gas law test and
  the real-time 10000-molecule test pass
- [x] 4.2 Document groups, pair interactions, and boundaries in the user guide with runnable examples, and verify the
  documentation test passes
- [x] 4.3 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass

## Workflow follow-up

- Archive the change after review, before applying the changes that build on groups.
