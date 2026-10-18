# Tasks

## 1. Lattices

- [ ] 1.1 Add `LatticeSpace`, `Neighborhood`, and lattice cell groups, and verify the Moore neighbourhood scenario and
  fixed and reflecting edges

## 2. Local rules

- [ ] 2.1 Add `LocalRule` with functions, tables, Wolfram numbers, and totalistic rules, and the lattice form and
  stepper on NumPy, and verify the rule 30 and glider scenarios
- [ ] 2.2 Add the schedules and stochastic rules, and verify the synchronous and reproducible scenarios
- [ ] 2.3 Fuse lattice rules on the Numba backend, and verify the large Game of Life scenario as a benchmark

## 3. Relaxation and Monte Carlo

- [ ] 3.1 Add `Relaxation` with events stored in trajectories, and verify the sandpile scenario
- [ ] 3.2 Add `Energy`, `Move`, and `Sampler` with symbolic energy changes and checkerboard sweeps, and verify the
  Ising transition scenario

## 4. Visualization and library

- [ ] 4.1 Log lattices to Rerun as images colored by state, and verify on an in-memory recording
- [ ] 4.2 Add the lattice models as registered factories, and verify the traffic scenario and the conformance suite

## 5. Documentation and gate

- [ ] 5.1 Document lattices, rules, schedules, relaxation, and Monte Carlo in the user guide, and verify the
  documentation test passes
- [ ] 5.2 Run formatting, checks, the docs build, and tests through nox, and the benchmarks with `--benchmarks`, and
  verify all of them pass
