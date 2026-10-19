# Spec Delta

## ADDED Requirements

### Requirement: PyClaw is an optional extra
The PyClaw backend SHALL be available only through an optional `clawpack` extra, and the default install SHALL NOT
need PyClaw. Asking for the PyClaw backend without the extra SHALL raise a named error before the first frame that
names the extra, the conda-forge package, and the in-house solver that runs without it.

#### Scenario: Default install
- **WHEN** the package is installed without extras and a library finite-volume model runs with its default solver
- **THEN** the run completes on the in-house path, and the solver reports the native backend

#### Scenario: Missing extra
- **WHEN** a user runs Burgers' equation with the PyClaw backend and `clawpack` is not installed
- **THEN** a named error is raised that names the `clawpack` extra, the conda-forge package, and the native solver

### Requirement: PyClaw runs finite-volume fields from their flux laws
The PyClaw backend SHALL run the conservation laws of lattice finite-volume fields from the same flux laws as the
native path, stepping frame by frame, and SHALL return trajectories with the same names and shapes. It SHALL reject
every element, field, or law it cannot run before the first frame.

#### Scenario: Trajectory from PyClaw
- **WHEN** the shallow-water dam break runs with the PyClaw backend
- **THEN** its trajectory returns `channel.h` and `channel.q` with the same shapes and frame times as a native run

#### Scenario: Unsupported problem
- **WHEN** a field on a triangular mesh or a group of agents is given to the PyClaw backend
- **THEN** a named error is raised that names the backend and each unsupported item

### Requirement: Flux laws lower to Riemann solvers with waves
The PyClaw backend SHALL map fluxes of a known form to PyClaw's shipped solvers with their parameters. It SHALL
generate a Roe-type wave solver from the eigen-decomposition of the flux Jacobian for other fluxes of up to three
states, and an HLL solver when no closed form exists. It SHALL report the solver it chose for every field.

#### Scenario: Known fluxes
- **WHEN** Burgers' equation and the shallow-water dam break run with the PyClaw backend
- **THEN** each field reports a shipped solver, and the gravity of the dam break reaches the solver as a parameter

#### Scenario: Generated Roe solver
- **WHEN** Burgers' equation on 1600 cells and the dam break on 1600 cells run with the solver forced to the generated
  Roe solver and then with the shipped one, at the same limiter and Courant number
- **THEN** the L1 error of the generated solver against a reference on 8 times as many cells is at most 1.5 times the
  error of the shipped solver

#### Scenario: HLL fallback
- **WHEN** a field runs a flux whose Jacobian has no closed-form eigenvalues on a periodic lattice with PyClaw
- **THEN** the field reports the HLL solver, the totals stay constant within 1e-12 relative error, and no new extrema
  form at shocks

### Requirement: PyClaw applies the field boundaries
The PyClaw backend SHALL apply the periodic, zero-flux, and fixed boundaries of every state of a field, including
different boundaries for different states, with the same ghost-cell values as the native path.

#### Scenario: Periodic conservation
- **WHEN** Burgers' equation runs on a periodic domain with the PyClaw backend until after the shock forms
- **THEN** the total stays constant within 1e-12 relative error

#### Scenario: Mixed boundaries
- **WHEN** a two-state traffic field with a fixed inflow density and a zero-flux discharge runs on both paths
- **THEN** the two solutions agree within the parity tolerance next to both edges

### Requirement: PyClaw applies sources and live parameters
The PyClaw backend SHALL apply rate laws on the conserved states as sources by operator splitting, and SHALL apply a
parameter change before the next frame.

#### Scenario: Damped Burgers
- **WHEN** Burgers' equation with the source `-k u` runs on a periodic domain with the PyClaw backend
- **THEN** the total decays as `exp(-k t)` within 1e-4 relative error

#### Scenario: Gravity change
- **WHEN** a user doubles the gravity of the dam break between two frames of a PyClaw run
- **THEN** the shock runs faster in the following frames, as in a native run with the same change

### Requirement: PyClaw agrees with the native solver
The PyClaw backend and the native solver SHALL produce the same solution within the discretisation error from the same
field laws, and the difference SHALL shrink as the lattice is refined.

#### Scenario: Backend parity
- **WHEN** Burgers' equation and the dam break run on both paths at 800 and 1600 cells
- **THEN** the L1 distance between the two paths is below 5e-3 at 1600 cells and smaller than at 800 cells

### Requirement: PyClaw is faster and as accurate as the native solver
The PyClaw backend SHALL run lattice finite-volume fields at least an order of magnitude faster than the native solver
with no larger error, measured on the same machine after the first call.

#### Scenario: Burgers speed
- **WHEN** Burgers' equation on 3200 cells runs to time 2 on both paths
- **THEN** the PyClaw run is at least 20 times faster, and its L1 error against a reference on 8 times as many cells is
  no larger than the native error

#### Scenario: Dam break speed
- **WHEN** the dam break on 1600 cells runs to time 0.5 on both paths
- **THEN** the PyClaw run is at least 50 times faster, and its L1 error against the exact Riemann solution is no larger
  than the native error
