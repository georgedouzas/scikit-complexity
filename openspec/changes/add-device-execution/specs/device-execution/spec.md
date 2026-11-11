# Spec Delta

## Purpose

Lets a user run a compiled problem on a chosen device, a CPU or a GPU, in a chosen precision, with every internal step
of a frame on the device and noise and jumps drawn there.

## ADDED Requirements

### Requirement: The JAX backend runs on a chosen device
`JaxBackend` SHALL take `device`, one of `'cpu'`, `'gpu'`, `'cuda'`, `'rocm'`, a `jax.Device`, or `None` for the
default device of JAX, and SHALL run every traced function and loop of a problem on that device. An unknown name or a
platform without a device SHALL raise `BackendError`, whose message lists the devices JAX sees and the install command
of the GPU plugin.

#### Scenario: CPU device
- **WHEN** a Kuramoto system of 1000 oscillators runs with `JaxBackend(device='cpu')` for 10 frames
- **THEN** its final state agrees with the run on the NumPy backend within 1e-12 relative error, and the problem
  records the backend `'jax:cpu'`

#### Scenario: Missing GPU
- **WHEN** a user asks for `JaxBackend(device='gpu')` on a machine where JAX sees no GPU
- **THEN** preparing the backend raises `BackendError`, whose message lists the CPU device and names `jax[cuda13]`

#### Scenario: GPU device
- **WHEN** the same Kuramoto run uses `JaxBackend(device='gpu')` on a machine with a GPU, in a test marked `gpu`
- **THEN** its final state agrees with the CPU run within 1e-10 relative error

### Requirement: The JAX backend takes a precision
`JaxBackend` SHALL take `precision`, `'float64'` by default or `'float32'`, and SHALL compute every function and loop
of a problem in that precision.

#### Scenario: Single precision
- **WHEN** exponential decay runs with `RK4Solver` and step 0.01 to time 1 with `precision='float32'`
- **THEN** the state is a float32 array, and it agrees with `exp(-1)` within 1e-6 relative error

#### Scenario: Unknown precision
- **WHEN** a user passes `precision='float16'`
- **THEN** preparing the backend raises `ValidationError`, whose message names `'float64'` and `'float32'`

### Requirement: Frames run on the device
The JAX backend SHALL fill the loops `iterate`, `rk4`, `velocity_verlet`, `yoshida4`, `euler_maruyama`, and
`gillespie` of `AcceleratedSpec` with compiled JAX loops, and a run on it SHALL copy the state from the device to the
host only when a frame is recorded.

#### Scenario: One copy per frame
- **WHEN** a Kuramoto system runs with `JaxBackend(device='cpu')`, `RK4Solver`, step 0.001, and 10 frames 0.1 apart
- **THEN** the run calls the RK4 loop 10 times and transfers 10 states to the host, not 1000

#### Scenario: Same values as the forward functions
- **WHEN** a logistic map, a Lorenz system, and a harmonic oscillator under `VelocityVerletSolver` run on the JAX
  backend with loops and with the forward functions alone
- **THEN** their trajectories agree within 1e-12 relative error

### Requirement: Noise and jumps run on the device
The JAX backend SHALL lower noise laws of every noise kind except fractional noise, and jump laws by the direct method
of Gillespie, drawing from a JAX key derived from the random state of the run. Fractional noise SHALL raise
`BackendError` that names the NumPy and Numba backends.

#### Scenario: Geometric Brownian motion in distribution
- **WHEN** 2000 seeded runs of geometric Brownian motion by Euler-Maruyama with step 0.01 to time 1 execute on the JAX
  backend and on the NumPy backend
- **THEN** the means and the variances of the final values agree within three standard errors

#### Scenario: Stochastic SIR on the device
- **WHEN** 500 seeded runs of a stochastic SIR reaction network of 1000 people run with `GillespieSolver` on the JAX
  backend and on the Numba backend
- **THEN** the mean final sizes agree within three standard errors

#### Scenario: Fractional noise
- **WHEN** a system with fractional noise is compiled with the JAX backend
- **THEN** it raises `BackendError`, whose message names the NumPy and Numba backends

### Requirement: Device runs are benchmarked against the Numba backend
The benchmarks SHALL time device runs against the Numba backend on the same machine, with targets written as ratios.

#### Scenario: Large Kuramoto system on the CPU device
- **WHEN** a Kuramoto system of 1000000 oscillators runs 100 frames with `RK4Solver` and step 0.01 on the Numba
  backend and on `JaxBackend(device='cpu')`, in a test marked `benchmark`
- **THEN** the JAX run takes at most twice the Numba time

#### Scenario: Large Kuramoto system on the GPU
- **WHEN** the same run uses `JaxBackend(device='gpu')`, in a test marked `benchmark` and `gpu`
- **THEN** the JAX run takes at most a fifth of the Numba time on all cores of the same machine
