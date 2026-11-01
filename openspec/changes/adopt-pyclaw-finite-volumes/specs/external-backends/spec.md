# Spec Delta

## ADDED Requirements

### Requirement: PyClaw is an optional extra
The PyClaw solver SHALL be available only through an optional `clawpack` extra, and the default install SHALL NOT need
PyClaw. Passing the PyClaw solver without the extra SHALL be reported in the compile report, before compilation, as an
error whose fix names the extra, the conda-forge package, and the in-house solver that runs without it.

#### Scenario: Default install
- **WHEN** the package is installed without extras and a library finite-volume model runs with its default solver
- **THEN** the run completes on the in-house path, and the solver reports the native path

#### Scenario: Missing extra
- **WHEN** a user checks a simulation of Burgers' equation with `PyClawSolver` and `clawpack` is not installed
- **THEN** `Simulation.check` raises a `ValidationError` whose one issue has the code `solver-unavailable` and a fix
  that names the `clawpack` extra, the conda-forge package, and `SSPRKSolver`

### Requirement: PyClaw runs finite-volume fields from their compiled kernels
The PyClaw solver SHALL run the conservation laws of finite-volume fields on lattices from the same compiled
`FiniteVolumeKernel`s as the native path, stepping frame by frame, and SHALL return trajectories with the same names
and shapes. It SHALL declare that it supports continuous time, lattice spaces, and flux, rate, and boundary laws, and
SHALL reject every field or law it cannot run before the first frame.

#### Scenario: Trajectory from PyClaw
- **WHEN** the shallow-water dam break runs with the PyClaw solver
- **THEN** its trajectory returns the depth and the discharge of the channel with the same shapes and frame times as a
  native run

#### Scenario: Unsupported problem
- **WHEN** a field on a triangular mesh or a variable group with rule laws is checked with the PyClaw solver
- **THEN** the report holds one `solver-unsupported` issue per unsupported item, each naming the solver

### Requirement: Flux systems lower to Riemann solvers with waves
The PyClaw solver SHALL map flux systems of a known form with uniform parameters to PyClaw's shipped solvers with their
parameters. It SHALL generate a Roe-type wave solver from the eigen-decomposition of the flux Jacobian for other flux
systems of up to three states and for flux systems that read per-cell parameters, and an HLL solver when no closed form
exists. It SHALL report the solver it chose for every field.

#### Scenario: Known fluxes
- **WHEN** Burgers' equation and the shallow-water dam break run with the PyClaw solver
- **THEN** each field reports a shipped solver, and the gravity of the dam break reaches the solver as a parameter

#### Scenario: Generated Roe solver
- **WHEN** Burgers' equation on 1600 cells and the dam break on 1600 cells run with the solver forced to the generated
  Roe solver and then with the shipped one, at the same limiter and Courant number
- **THEN** the L1 error of the generated solver against a reference on 8 times as many cells is at most 1.5 times the
  error of the shipped solver

#### Scenario: Per-cell parameters
- **WHEN** the Payne traffic model, whose flux system reads per-cell values, runs with the PyClaw solver and with
  `SSPRKSolver` on 800 and 1600 cells
- **THEN** its field reports the generated Roe solver, and the L1 distance between the two paths is smaller at 1600
  cells than at 800 cells

#### Scenario: HLL fallback
- **WHEN** a field runs a flux whose Jacobian has no closed-form eigenvalues on a periodic lattice with PyClaw
- **THEN** the field reports the HLL solver, the totals stay constant within 1e-12 relative error, and no new extrema
  form at shocks

### Requirement: PyClaw applies the field boundaries
The PyClaw solver SHALL apply the periodic, zero-flux, and fixed boundaries of every state of a field, including
different boundaries for different states, with the same ghost-cell values as the native path.

#### Scenario: Periodic conservation
- **WHEN** Burgers' equation runs on a periodic domain with the PyClaw solver until after the shock forms
- **THEN** the total stays constant within 1e-12 relative error

#### Scenario: Mixed boundaries
- **WHEN** a two-state traffic field with a fixed inflow density and a zero-flux discharge runs on both paths
- **THEN** the two solutions agree within the parity tolerance next to both edges

### Requirement: PyClaw applies sources and live parameters
The PyClaw solver SHALL apply the source rates of the conserved states, from the `source` of a conservation law or
from other rate laws, by operator splitting, and SHALL apply a parameter change before the next frame.

#### Scenario: Damped Burgers
- **WHEN** Burgers' equation with the source `-k u` of its `ConservationInteraction` runs on a periodic domain with the
  PyClaw solver
- **THEN** the total decays as `exp(-k t)` within 1e-4 relative error

#### Scenario: Gravity change
- **WHEN** a user doubles the gravity of the dam break between two frames of a PyClaw run
- **THEN** the shock runs faster in the following frames, as in a native run with the same change

### Requirement: PyClaw agrees with the native solver
The PyClaw solver and `SSPRKSolver` SHALL produce the same solution within the discretisation error from the same field
laws, and the difference SHALL shrink as the lattice is refined.

#### Scenario: Backend parity
- **WHEN** Burgers' equation and the dam break run on both paths at 800 and 1600 cells
- **THEN** the L1 distance between the two paths is below 5e-3 at 1600 cells and smaller than at 800 cells

### Requirement: PyClaw is faster and as accurate as the native solver
The PyClaw solver SHALL run lattice finite-volume fields faster than `SSPRKSolver` on the current kernel with no larger
error, measured on the same machine after the first call, by at least half of the ratio the re-measurement of this
change records.

#### Scenario: Burgers speed
- **WHEN** Burgers' equation on 3200 cells runs to time 2 on both paths
- **THEN** the PyClaw run is faster by at least half of the recorded ratio, and its L1 error against a reference on 8
  times as many cells is no larger than the native error

#### Scenario: Dam break speed
- **WHEN** the dam break on 1600 cells runs to time 0.5 on both paths
- **THEN** the PyClaw run is faster by at least half of the recorded ratio, and its L1 error against the exact Riemann
  solution is no larger than the native error
