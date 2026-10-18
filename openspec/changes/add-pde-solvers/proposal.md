# Proposal

## Why

`add-spatial-fields` discretises fields by finite differences and integrates them with the existing ODE solvers. That
covers moderate reaction-diffusion models, but the references go further:
- **Spectral methods:** Datseris and Parlitz (2022 ch. 11) solve pattern-forming PDEs spectrally, which is far more
  accurate per grid point on periodic domains.
- **Stiff dynamics:** Gros (2015 ch. 4) and Trzaska (2023 ch. 4) study stiff chemical and pattern dynamics that explicit
  steppers handle only with tiny steps.
- **Turbulence and convection:** Satz (2022 ch. 12) and Mihailović et al. (2023 ch. 8) discuss turbulence and
  Rayleigh-Bénard convection, which need incompressible flow solvers.
- **Conservation laws:** shocks and fronts need finite-volume schemes that conserve and do not oscillate.
- **Irregular domains:** domains that are not grids need finite elements.

## What Changes

- Add pseudo-spectral discretisation of fields on periodic domains. Operators act in Fourier space, nonlinear terms in
  real space, and aliasing is removed with the two-thirds rule.
- Add stiff and exponential steppers: implicit-explicit Euler and second-order schemes, Crank-Nicolson, and the
  fourth-order exponential time-differencing Runge-Kutta scheme ETDRK4. The compiler splits the linear stiff part from
  the nonlinear part of a field's laws symbolically.
- Add finite-volume discretisation for conservation laws: fluxes declared per field, Godunov and Rusanov fluxes,
  MUSCL reconstruction with slope limiters, and conservative updates.
- Add finite elements on unstructured triangular and tetrahedral meshes for diffusion and reaction-diffusion, through
  scikit-fem, with meshes from simple shapes or files.
- Add incompressible two-dimensional flow in vorticity-streamfunction form on periodic domains, and Boussinesq
  convection.
- Add library models:
  - Kuramoto-Sivashinsky;
  - the complex Ginzburg-Landau equation;
  - Swift-Hohenberg patterns;
  - Burgers' shock;
  - the shallow-water dam break;
  - decaying two-dimensional turbulence;
  - Rayleigh-Bénard convection;
  - heat on an L-shaped domain with finite elements.

## Capabilities

### New Capabilities

- `pde-solvers`: spectral and finite-volume discretisations, finite elements, stiff and exponential steppers, and
  incompressible flow.

### Modified Capabilities

- `model-library`: PDE models.

## Impact

- `skcomplex.model` gains `discretization=` on fields with `'finite-difference'`, `'spectral'`, `'finite-volume'`, and
  `'finite-element'`, plus `Flux` and `Mesh`. `skcomplex.solvers` gains `IMEXEuler`, `IMEX2`, `CrankNicolson`, and
  `ETDRK4`.
- New dependency `scikit-fem` for finite elements, and SciPy's FFT for spectral fields.
- Depends on `add-spatial-fields` and `add-analysis-toolkit` for spectra in the acceptance tests.
