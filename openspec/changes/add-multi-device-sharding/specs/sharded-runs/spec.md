# Spec Delta

## Purpose

Lets a user spread a batch of runs, or one system too large for a device, over the devices of a node and of a cluster,
with the same results as on one device.

## ADDED Requirements

### Requirement: The JAX backend takes several devices
`JaxBackend` SHALL take a sequence of devices as `device`, and SHALL run every problem over a mesh of those devices. In
a run of several processes, a sequence that does not span the devices of every process SHALL raise `BackendError`,
whose message names `jax.devices()`.

#### Scenario: Emulated devices
- **WHEN** a Kuramoto system of 4000 oscillators runs on `JaxBackend(device=jax.devices('cpu'))` with four emulated
  CPU devices
- **THEN** its trajectory equals the run on one device within 1e-10

#### Scenario: Devices of one process in a cluster run
- **WHEN** a script run as two processes passes `jax.local_devices()` to the backend
- **THEN** preparing the backend raises `BackendError`, whose message names `jax.devices()`

### Requirement: Batches split over devices
A batch of runs on several devices SHALL split its run axis over them, and its results SHALL equal those on one device.

#### Scenario: Ensemble on four devices
- **WHEN** a vectorised ensemble of 1000 stochastic SIR runs executes on four emulated CPU devices
- **THEN** the result equals the result on one device within 1e-10

#### Scenario: Scaling over GPUs
- **WHEN** a vectorised ensemble of 100000 stochastic SIR runs executes on every GPU of a node and on one of them, in a
  test marked `benchmark` and `gpu`
- **THEN** the run on every GPU is at least 0.8 times the GPU count faster

### Requirement: One system splits over devices
A single run on several devices SHALL split every group of at least 1000 members along its members, SHALL exchange
halos of lattices between neighbouring devices, and SHALL give the result of the run on one device.

#### Scenario: Lattice on four devices
- **WHEN** an Ising system of 512 by 512 sites runs 100 checkerboard sweeps with random state 0 on four emulated CPU
  devices
- **THEN** its spins equal those of the run on one device

#### Scenario: Network and flock on four devices
- **WHEN** a quenched mean field SIS system on a network of 10000 nodes and a Vicsek system of 10000 particles run on
  four emulated CPU devices
- **THEN** their trajectories equal the runs on one device within 1e-10

#### Scenario: Weak scaling of a lattice
- **WHEN** an Ising system of 8192 by 8192 sites runs 100 sweeps on one GPU, and one of 8192 times the GPU count by
  8192 sites on every GPU of a node, in a test marked `benchmark` and `gpu`
- **THEN** the second run takes at most 1.25 times the first

### Requirement: Systems span several processes
With one process per node, each calling `jax.distributed.initialize()`, a backend given `jax.devices()` SHALL run one
system or one batch over the devices of every process, and every process SHALL hold the same observables.

#### Scenario: Two CPU processes
- **WHEN** two local processes with two emulated CPU devices each run the same Ising script with the gloo collectives,
  in a test marked `slow`
- **THEN** both processes hold the same magnetisation series, equal to the run of one process on one device
