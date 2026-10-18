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
