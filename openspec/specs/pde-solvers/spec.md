# pde-solvers Specification

## Purpose

Lets a user discretise the same field model spectrally, by finite volumes, or by finite elements, and integrate stiff
and turbulent dynamics with implicit and exponential steppers.

## Requirements

### Requirement: Fields discretise spectrally on periodic domains
Fields on periodic lattices SHALL discretise spectrally with dealiasing, from the same symbolic laws as finite
differences.

#### Scenario: Spectral accuracy
- **WHEN** a smooth periodic profile diffuses on 64 points spectrally and with finite differences
- **THEN** the spectral error against the exact solution is at least 1000 times smaller

### Requirement: Stiff fields use implicit and exponential steppers
The framework SHALL split a field's laws into a linear stiff part and a nonlinear remainder, and integrate them with
implicit-explicit, Crank-Nicolson, and fourth-order exponential time-differencing steppers. A multistep stepper SHALL
restart its history whenever a call does not continue the state, time, and step of its previous call.

#### Scenario: Kuramoto-Sivashinsky
- **WHEN** the Kuramoto-Sivashinsky equation runs spectrally with ETDRK4 at steps 0.25 and 0.125 on 128 modes
- **THEN** the run is stable at both steps, and the error between them shrinks at fourth order before chaos amplifies it

#### Scenario: Second call from the same state
- **WHEN** `IMEX2` advances the Kuramoto-Sivashinsky equation from its initial state to time 0.1 twice in a row
- **THEN** both calls return the same state, equal to that of a fresh `IMEX2` advanced over the same interval

### Requirement: Conservation laws use finite volumes
Fields with declared fluxes SHALL discretise by finite volumes with limited reconstruction, conserving their totals and
resolving shocks without spurious oscillations.

#### Scenario: Burgers shock
- **WHEN** Burgers' equation develops a shock from a sine wave on a periodic domain
- **THEN** the total stays constant within 1e-12 relative error, and the solution has no new extrema near the shock

### Requirement: Fields run on unstructured meshes
Fields SHALL discretise by finite elements on triangular and tetrahedral meshes for diffusion and reaction terms, and
stream to Rerun as colored meshes.

#### Scenario: L-shaped steady state
- **WHEN** heat diffuses on an L-shaped mesh with fixed boundary temperatures until steady
- **THEN** the solution converges at second order in the mesh size away from the re-entrant corner

### Requirement: Incompressible flow runs in two dimensions
The framework SHALL simulate incompressible two-dimensional flow in vorticity-streamfunction form, with optional
Boussinesq buoyancy.

#### Scenario: Energy decay
- **WHEN** decaying turbulence runs from a random vorticity field without forcing
- **THEN** the kinetic energy decreases monotonically, and the enstrophy decreases faster than the energy

#### Scenario: Convection onset
- **WHEN** Rayleigh-Bénard convection runs with free-slip walls in a domain of width `2 sqrt(2)` times its height, at
  Rayleigh numbers 500 and 1000
- **THEN** perturbations decay at 500 and grow into convection rolls at 1000, around the critical value
  `27 π⁴ / 4 ≈ 657.5`

### Requirement: SSPRK steps respect diffusion
The strong-stability-preserving Runge-Kutta stepper SHALL limit its step by the diffusive stability limit of the
Laplacian rates of finite-volume fields, for the current parameter values, as well as by the CFL condition of their
fluxes.

#### Scenario: Viscous Burgers
- **WHEN** viscous Burgers' equation with viscosity 0.5 runs on 200 finite-volume cells of a periodic interval of
  length 2π with `SSPRK` and frames of 0.05 until time 0.5
- **THEN** the solution stays within the range of its initial values, and the run raises no stability warning
