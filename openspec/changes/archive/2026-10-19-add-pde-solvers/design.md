# Design

## Context

Fields from `add-spatial-fields` hold one value per site, operators stay symbolic until lowering, and the method of
lines turns them into rate laws. Different discretisations can therefore share the symbolic laws and change only the
lowering and the stepper.

## Goals / Non-Goals

**Goals:**

- One symbolic model, discretised by finite differences, spectral methods, finite volumes, or finite elements by
  changing one argument where the method applies.
- Stiff and turbulent models at grid sizes that the references use, in real time where practical.

**Non-Goals:**

- Three-dimensional Navier-Stokes, compressible flow, and adaptive mesh refinement.
- GPU backends. The spectral path keeps array operations that a GPU backend could adopt later.

## Decisions

### Spectral fields

With `discretization='spectral'`, fields on periodic lattices store real-space values. Operators lower to
multiplication by wavenumber arrays after `rfftn`, nonlinear products are computed in real space, and the two-thirds
rule zeroes the top third of wavenumbers. The symbolic laws are unchanged.

### Splitting for stiff steppers

The compiler splits each field's rate into a linear part `L u` with constant coefficients and a remainder `N(u)` by
symbolic differentiation and collection of linear operator terms. `ETDRK4` computes its coefficients for the diagonal
`L` in Fourier space by contour integrals (Kassam and Trefethen). The IMEX steppers treat `L` implicitly with sparse
solves on finite-difference grids and diagonal solves in Fourier space.

### Finite volumes

`Flux(field, expression)` declares the physical flux, and `discretization='finite-volume'` updates cell averages by
flux differences. Interface states come from MUSCL reconstruction with minmod or van Leer limiters, and the numerical
flux is Rusanov by default or Godunov where an exact Riemann solver is given. The stepper is strong-stability-preserving
RK2 or RK3, with the step chosen from the CFL condition.

### Finite elements

`Mesh.rectangle`, `Mesh.l_shape`, `Mesh.disk`, and `Mesh.load(path)` build meshes with scikit-fem. With
`discretization='finite-element'`, diffusion and reaction terms of fields are assembled into mass and stiffness
matrices once, and the reaction terms are evaluated at nodes. Time stepping uses the IMEX steppers with the mass
matrix. Fields on meshes stream to Rerun as triangle meshes colored by value.

### Incompressible flow

`incompressible_flow(nu, forcing=None, buoyancy=None)` builds a vorticity field on a periodic grid. Its streamfunction
comes from a spectral Poisson solve, and advection uses the streamfunction velocity. Boussinesq buoyancy couples a
temperature field for convection.

## Risks / Trade-offs

- [Splitting fails for non-constant coefficients] → Such terms stay in `N`, so the scheme remains correct with a
  smaller stable step, and the compiler reports what it treated implicitly.
- [New dependency] → scikit-fem is pure Python on NumPy and SciPy, and only finite elements import it.

## Implementation Notes

- `Field(..., discretization=...)` selects the discretization of the same laws. Spectral operators and the finite
  element Laplacian are global, so the compiler lowers them to derived reads: the kernel evaluates the operator's
  operand over every site, then applies a Fourier multiplier or a sparse matrix. Chains of spectral operators on one
  operand compose into one multiplier, and every spectral multiplier is dealiased by the two-thirds rule, so
  nonlinear terms written inside operators, such as `divergence([u**2 / 2])`, are dealiased. Derived reads run on
  NumPy, and the Numba backend falls back for them per law kind.
- The splitting collects, from every rate of a field as written, the terms `coefficient * op(op(... state))` with
  coefficients of parameters only, couplings between states of one field included. Solvers take the remainder as
  `N(x) = f(x) - L x`, which is exact whatever the linear part holds, so terms left out of `L` only cost stability.
  The linear part is a small matrix per Fourier mode for spectral fields and a sparse matrix for finite differences
  and finite elements, built from the same neighbour tables and mesh matrices as the rates.
- ETDRK4 computes its coefficients by contour integrals over a full circle around every eigenvalue of the per-mode
  matrices, since coupled fields such as Ginzburg-Landau and convection have complex eigenvalues, where the
  half-circle of Kassam and Trefethen holds for real ones only. It is fourth order on Kuramoto-Sivashinsky until the
  chaos amplifies the error: the error ratio between steps 0.25 and 0.125 is 16 at time 1 and falls to about 7 by time
  10. ETDRK4 needs spectral fields, and `IMEXEuler`, `IMEX2`, and `CrankNicolson` take every discretization.
- Finite volumes add the flux differences of `Flux` laws to the drift, with MUSCL reconstruction, minmod or van Leer
  limiters, Rusanov fluxes whose wave speeds are the spectral radii of the symbolic flux Jacobians, and exact Godunov
  fluxes for one state from the symbolic critical points of the flux. `SSPRK` steps them by the CFL condition. The
  Burgers shock keeps its total to 1e-14 and forms no new extrema.
- Finite elements assemble linear elements with scikit-fem, which only meshes import, and use the lumped mass, so
  the semi-discrete Laplacian is `-M^-1 K`. Fixed boundary values, numbers or functions of the coordinates, hold the
  boundary nodes, whose rates and rows of the linear part are zero. Away from the re-entrant corner, the steady state
  of a harmonic boundary temperature converges at second order. A quadratic harmonic is exact at the nodes of these
  meshes, so the test uses `exp(x) sin(y)`.
- Incompressible flow is a spectral vorticity field whose velocity comes from `inverse_laplacian`, with advection in
  conservative form so that it is dealiased. Convection between free-slip walls holds the layer on a doubled periodic
  lattice with odd states. Rounding breaks that symmetry and lets the elevator modes of periodic convection grow
  without bound, so fields take a `symmetry` option that projects the rates onto their odd or even parts, which keeps
  the symmetry exact. With it, perturbations decay at Rayleigh number 500 and saturate into steady rolls at 1000.
- Lattices take one spacing per axis, which the convection layer of width `2 sqrt(2)` and height 1 needs.
- `incompressible_flow` lives with the fluid models in `skcomplex.models.fluids`, beside the factories built on it.
