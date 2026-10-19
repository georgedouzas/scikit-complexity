# Tasks

## 1. Discretisations

- [x] 1.1 Add spectral fields with dealiasing, and verify the spectral accuracy scenario
- [x] 1.2 Add finite volumes with fluxes, limiters, and SSP steppers, and verify the Burgers scenario
- [x] 1.3 Add meshes and finite elements through scikit-fem with Rerun mesh logging, and verify the L-shaped scenario

## 2. Steppers

- [x] 2.1 Add the symbolic linear splitting and the IMEX, Crank-Nicolson, and ETDRK4 steppers, and verify the
  Kuramoto-Sivashinsky scenario

## 3. Flow

- [x] 3.1 Add incompressible two-dimensional flow with Boussinesq buoyancy, and verify the energy decay and convection
  onset scenarios

## 4. Library

- [x] 4.1 Add the PDE models as registered factories, and verify the Swift-Hohenberg scenario and the conformance suite

## 5. Documentation and gate

- [x] 5.1 Add scikit-fem to the dependencies, document discretisations, steppers, and flow in the user guide, and
  verify the documentation test passes
- [x] 5.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
