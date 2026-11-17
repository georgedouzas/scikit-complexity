# Tasks

## 1. Prerequisite and baselines

- [ ] 1.1 Check that `add-device-execution` is applied, and stop if it is not. Time the five probes of the design on
  the Numba backend of the current code, write the times and the machine in the benchmark docstrings, and verify they
  repeat within 10% over three runs

## 2. Lattices

- [ ] 2.1 Build the table of neighbour indices of a `LatticeSpace` and its neighbourhood on the host, lower lattice
  reads to gathers in `_jax.py`, and verify a lattice drift agrees with the Numba backend within 1e-12 on square,
  hexagonal, and bounded lattices in `tests/core/engine/backends/test_jax.py`
- [ ] 2.2 Add `lattice` to `AcceleratedSpec`, fill it in `_jax_loops.py` for local rules and array updates with the
  synchronous, checkerboard, and colored schedules, make `LatticeSolver` read it when a backend fills it, and verify an
  elementary automaton and the contact process against the Numba backend
- [ ] 2.3 Add checkerboard and colored Monte Carlo updates to the `lattice` loop, make `MonteCarloSolver` read it, and
  verify the Ising-magnetisation scenario, marked slow
- [ ] 2.4 Raise `BackendError` for the random schedules, and verify the random-sequential-updates scenario

## 3. Neighbour search

- [ ] 3.1 Add `core/engine/backends/_jax_neighbours.py` with the cell list in periodic and bounded boxes and radius
  neighbours, and verify the radius part of the same-neighbours scenario in
  `tests/core/engine/backends/test_jax_neighbours.py`
- [ ] 3.2 Add cones and nearest neighbours, and verify the rest of the same-neighbours scenario
- [ ] 3.3 Add `capacity` to `JaxBackend` with the automatic value and the overflow flag, and verify the overflow
  scenario
- [ ] 3.4 Lower neighbour reads and reductions of `RadiusNeighborhood` and `NearestNeighborhood` to the cell list in
  the maps, ODEs, and velocity Verlet loops, and verify the ordered-flock scenario and a Lennard-Jones fluid of 1000
  particles against the Numba backend within 1e-10 over 100 steps

## 4. Networks and meshes

- [ ] 4.1 Lower reads and reductions of `AdjacencyNeighborhood` to `segment_sum` over the edge lists, and verify the
  same-drift-as-the-Numba-backend and epidemic-threshold scenarios
- [ ] 4.2 Lower the explicit operators of mesh fields to sums over the nonzero entries of their matrices, raise
  `BackendError` for implicit steppers, and verify heat diffusion on a mesh against the Numba backend within 1e-10

## 5. Vectorised agent laws

- [ ] 5.1 Lower `TransitionLaw` and `SampleLaw` to draws per member, and verify the transitions-in-distribution
  scenario
- [ ] 5.2 Raise `BackendError` for `RuleLaw` and `MatchLaw`, and verify the agent-form scenario

## 6. Observables on the device

- [ ] 6.1 Trace the observables of a problem into one function on the device and call it from the recorder, and verify
  the observables of a Kuramoto run equal those of the NumPy backend within 1e-12
- [ ] 6.2 Add `record_states` to `simulate` and the recorder, and verify the order-parameter-of-a-large-flock and
  reading-a-state-that-was-not-recorded scenarios in `tests/core/engine/runtime/test_simulate.py`
- [ ] 6.3 Set `record_states=False` in experiments whose outputs are all observables, and verify the
  ensemble-of-observables scenario in `tests/core/experiments/test_runs.py`

## 7. Benchmarks

- [ ] 7.1 Add the benchmarks of the large Ising lattice, the million-particle flock, and the million-node network, and
  verify the CPU targets locally
- [ ] 7.2 Add the benchmarks of the Lennard-Jones fluid of 1000000 particles and of Gray-Scott on 2048 by 2048 sites
  with the targets of the design, and verify the CPU targets locally
- [ ] 7.3 Check that the GPU benchmarks carry the `gpu` marker, so the job `gpu` of `add-device-execution` runs them
  through `nox -s gpu` and they gate releases, and verify their targets in a manual run of `gpu.yml` on the GPU runner,
  or record in the design that no runner exists yet

## 8. Documentation

- [ ] 8.1 Add a large systems section to `docs/overview/user_guide/simulate.md` with the spaces and laws that run on
  the device, `capacity`, the schedules that run in parallel, and `record_states`, with a runnable Vicsek example on
  the CPU device, and verify the markdown tests run it
- [ ] 8.2 Update the docstring of `JaxBackend`, which says neighbourhood reductions have no JAX form, and verify the
  doctests pass

## 9. Integration

- [ ] 9.1 Run `nox` (formatting, ruff, mypy, pydoclint, import contracts, tests of the touched modules with the thread
  limits set to 1, doctests, and the strict docs build) and verify every session passes
- [ ] 9.2 Run `openspec validate add-large-system-kernels --strict` and verify it passes
