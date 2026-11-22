## 1. Seeds (#23, #24)

- [ ] 1.1 Rename `seed` to `random_state` in every ready-made system that has it, with its docstring saying it seeds
      what the system draws when prepared and that `simulate(random_state=)` seeds the dynamics
- [ ] 1.2 Give the same docstring sentence to every ready-made system that already has `random_state`
- [ ] 1.3 Update tests, doctests, docs, and gallery examples that pass `seed=`
- [ ] 1.4 Make `ensemble` set a spawned system seed per replicate when the system has `random_state`, with tests in
      `tests/core/experiments/test_ensemble.py`
- [ ] 1.5 Add a test that `VicsekSystem(random_state=4)` with a run seed is reproducible

## 2. Lattices and networks (#26, #27)

- [ ] 2.1 Choose the random sequential schedule for odd periodic lattices when the solver is `'auto'`, and name the
      `solver` setting in the checkerboard error, with tests mirroring the changed modules
- [ ] 2.2 Record derived states at frame 0 from the initial structure, with a test for `CoevolvingVoterSystem`

## 3. Analysis (#25, #29, #30, #31)

- [ ] 3.1 Sort fixed points by value and convert exact solutions with full precision, tested in
      `tests/core/analysis/test_stability.py`
- [ ] 3.2 Report period 0 for aperiodic orbits, tested in `tests/core/experiments/test_orbit_diagram.py`
- [ ] 3.3 Add `blocks`, `rule`, and `tie` to `coarse_grain`, tested in `tests/core/analysis/test_coarse_grain.py`
- [ ] 3.4 Show a normal form as a one line `DynamicalSystem` in the `stability` docstring and the guide

## 4. Performance (#32)

- [ ] 4.1 Profile `orbit_diagram` over 1200 values and remove the dominant cost, with a timing check kept out of the
      quick tests
- [ ] 4.2 Set the default `dt` of `CurieWeissSystem` to 1 and update its tests and docs

## 5. Docstrings (#28)

- [ ] 5.1 Remove the leftover note from `IdealGasSystem`
- [ ] 5.2 Describe `GalamMajoritySystem.share` as the probability that each agent starts in favour
- [ ] 5.3 Make the states listed by `BernoulliProcessSystem` match its trajectory names

## 6. Cleanup

- [ ] 6.1 Update the Project Profile of `openspec/constitution.md` to the current parts
- [ ] 6.2 Make `LoweredFunctionsSpec` private and update its uses and test

## 7. Close

- [ ] 7.1 Run the fast checks (ruff, black, mypy, lint-imports, conventions) and the tests of the changed modules
- [ ] 7.2 Close issues #23 to #32 with a comment naming the commit
