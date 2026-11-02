## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: SSPRK steps respect diffusion
The strong-stability-preserving Runge-Kutta stepper SHALL limit its step by the diffusive stability limit of the
Laplacian rates of finite-volume fields, for the current parameter values, as well as by the CFL condition of their
fluxes.

#### Scenario: Viscous Burgers
- **WHEN** viscous Burgers' equation with viscosity 0.5 runs on 200 finite-volume cells of a periodic interval of
  length 2π with `SSPRK` and frames of 0.05 until time 0.5
- **THEN** the solution stays within the range of its initial values, and the run raises no stability warning
