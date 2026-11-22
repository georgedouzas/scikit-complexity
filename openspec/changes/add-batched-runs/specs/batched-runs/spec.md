# Spec Delta

## Purpose

Lets a user run many runs of one compiled system as one batch, with a leading run axis on its states, parameters, and
random streams, on every backend, so ensembles and sweeps of any kind of system use a GPU or every core.

## ADDED Requirements

### Requirement: A batch runs many runs of one problem
The runtime SHALL run a batch of initial states, parameters, schedules, and random streams of one compiled problem with
a solver that has an accelerated loop, and SHALL return one trajectory per run. The runs of a solver without an
accelerated loop SHALL run one by one, and the experiment SHALL warn with `BackendWarning`, whose message names the
solver and the solvers that batch.

#### Scenario: Batch of initial states
- **WHEN** a Lorenz system runs as a batch of 8 initial states with `RK4Solver` on the NumPy backend
- **THEN** each of the 8 trajectories equals the run of its initial state alone within 1e-12

#### Scenario: Solver without a loop
- **WHEN** an ensemble of 10 runs of a system whose solver is `ScipyODESolver` executes with `vectorized=True`
- **THEN** it equals the result with `vectorized=False` and warns with `BackendWarning`, whose message names
  `ScipyODESolver` and `RK4Solver`

### Requirement: Batches run on every backend
The JAX backend SHALL run a batch as one vectorised device program, the Numba backend SHALL run it on threads with one
run per thread, and the NumPy backend SHALL run it one run after another as the reference.

#### Scenario: Backends agree on a batch
- **WHEN** a batch of 64 seeded runs of a stochastic logistic map runs on the NumPy, Numba, and JAX backends
- **THEN** the NumPy and Numba batches are equal, and the JAX batch equals the JAX runs one by one within 1e-10

### Requirement: Batches do not change results
A run in a batch SHALL draw from the same random stream as the run alone, so a result SHALL not depend on whether the
runs are batched or on the batch size.

#### Scenario: Batch sizes
- **WHEN** an ensemble of 100 stochastic SIR runs with seed 7 executes with `vectorized=False`, and with
  `vectorized=True` and batch sizes 1, 7, and 100, on the Numba backend
- **THEN** the four results are equal

### Requirement: Batch sizes fit the memory
With `batch_size=None` a batch SHALL hold as many runs as fit in half of the free memory of the device, or a quarter of
the physical memory of the host for the CPU, from the bytes one run needs. A batch that runs out of memory SHALL raise
`ExperimentError`, whose message names `batch_size`.

#### Scenario: Automatic size
- **WHEN** the free memory reported for the device is 1 GB and one run needs 1 MB
- **THEN** the batches hold 500 runs

#### Scenario: Too large a batch
- **WHEN** a user passes a batch size whose runs exceed the memory of the device
- **THEN** it raises `ExperimentError`, whose message names `batch_size`
