# Tasks

## 1. Prerequisites and dependencies

- [ ] 1.1 Check that `add-multi-device-sharding` is applied, so `simulate` takes `store`, and stop if it is not
- [ ] 1.2 Add `mpi4py>=4.1.0` to the runtime dependencies in `pyproject.toml` and the Project Profile with its release
  date and licence, and verify `pdm install` resolves on Linux and macOS and `mpiexec -n 2 python -c "from mpi4py
  import MPI"` runs with the MPICH of `mpich`
- [ ] 1.3 Add a fixture in `tests/core/engine/runtime/conftest.py` that runs a script under `mpiexec -n 4` in a
  subprocess and returns what each rank prints, and verify it reports four ranks
- [ ] 1.4 Time Gray-Scott on 4096 by 4096 sites for 100 RK4 frames on one rank of the current code, and write the time
  and the machine in the benchmark docstring

## 2. Slabs

- [ ] 2.1 Add `communicator` to `simulate`, compute the rows each rank owns, and check that every lattice has the same
  number of rows, and verify the lattices-of-different-heights scenario in
  `tests/core/engine/runtime/test_decomposition.py`
- [ ] 2.2 Compile each lattice of a rank with the shape of its slab plus its halo in
  `core/engine/compile/passes/_lattices.py`, evaluate numbers and functions of sites per slab, slice full arrays, and
  verify the memory-per-rank scenario

## 3. Halo exchange

- [ ] 3.1 Add `core/engine/runtime/_decomposition.py` with the exchange of halo rows by `Sendrecv`, wrap the drift and
  update functions of a decomposed problem, skip the accelerated loops, and verify the deterministic-field-on-four-ranks
  scenario
- [ ] 3.2 Wrap from the last rank to the first for periodic lattices and fill the outer halos for fixed and reflecting
  boundaries, and verify the periodic-boundary scenario and the same run with a fixed boundary
- [ ] 3.3 Exchange halos once per colour of checkerboard and colored schedules, and verify a checkerboard Monte Carlo
  sweep on four ranks keeps every spin in its sublattice order

## 4. Observables and streams

- [ ] 4.1 Reduce sums, means, minima, and maxima over lattice sites with `Allreduce`, raise `ValidationError` for other
  observables of sites, and verify the observable-that-does-not-decompose scenario
- [ ] 4.2 Give rank `r` the child `r` of the spawned random state, and verify the same-seed-and-ranks and
  Ising-magnetisation-on-four-ranks scenarios, marked slow
- [ ] 4.3 Write the owned rows of every frame to the store of `simulate`, and verify the store of a four-rank Gray-Scott
  run equals the frames of the run on one rank

## 5. Benchmarks

- [ ] 5.1 Add the weak scaling benchmark on four ranks of one machine, and verify its target locally
- [ ] 5.2 Run the weak scaling on 16 and 64 ranks of a cluster with `srun`, and record the times in the design, or
  record that no cluster was available

## 6. Documentation

- [ ] 6.1 Add a domain decomposition section to `docs/overview/user_guide/simulate.md` with `communicator`, the
  systems that decompose, the observables that reduce, the stores of frames, and an `mpiexec` and an `srun` recipe as
  code that the tests do not run, and verify the docs build

## 7. Integration

- [ ] 7.1 Run `nox` (formatting, ruff, mypy, pydoclint, deptry, import contracts, tests of the touched modules with the
  thread limits set to 1, doctests, and the strict docs build) and verify every session passes
- [ ] 7.2 Run `openspec validate add-domain-decomposition --strict` and verify it passes
