# Proposal

## Why

Field laws read only their own site and, through `laplacian`, `partial`, and `upwind`, their nearest neighbours.
The one global operator, `inverse_laplacian`, solves a periodic Poisson equation on spectral fields only. Three
models in the inventory need more than that:
- the Amari neural field integrates the firing rate over the whole domain against a lateral-inhibition kernel
  (2112.12147 §1.3.1);
- the quasi-geostrophic double gyre inverts potential vorticity into a streamfunction through a modified Helmholtz
  equation, `(laplacian - 1/Rd²) psi = q - beta y`, in a closed basin with walls (1910.00583 §III.6.1);
- the gas-kinetic-based traffic model reads density and velocity at an advanced position, a look-ahead distance that
  depends on the local velocity (cond-mat/0012229 §III.E.4).

These are general field operators, convolution, shifted reads, and screened Poisson inversion, so they belong in the
core, and the constitution requires them to be built on a mature library rather than recoded.

## What Changes

- Add convolution of a field expression with a kernel, `convolve(expression, kernel)`, where the kernel is a symbolic
  function of the displacement whose parameters are model parameters. It is sampled on the lattice and evaluated by
  FFT, circular on periodic lattices and truncated at fixed or zero-flux edges.
- Add shifted reads, `shifted(expression, displacement)`, which read a field expression at a physical displacement
  per axis. A displacement of whole sites reads the neighbour tables, and any other displacement, including one that
  depends on the states, interpolates and follows the field's boundary conditions.
- Add modified-Helmholtz inversion, `inverse_helmholtz(expression, screening)`, the solution of
  `(laplacian - screening²) psi = expression`, on spectral fields, on finite-difference lattices with periodic,
  zero-flux, or fixed edges, on networks, and on finite-element meshes. `inverse_laplacian` becomes its case with zero
  screening, so it now also works outside spectral fields.
- Let the stencil operators read the value of a nonlocal operator at neighbouring sites, so `gradient` and `advection`
  apply to an inverted streamfunction on a finite-difference field.
- Add library models: the Amari neural field, the quasi-geostrophic double gyre, and the gas-kinetic-based non-local
  traffic model.

## Capabilities

### New Capabilities

- `nonlocal-operators`: convolution kernels, shifted and look-ahead reads, and modified-Helmholtz inversion on fields,
  and their composition with the local operators.

### Modified Capabilities

- `model-library`: the Amari neural field, the quasi-geostrophic double gyre, and the gas-kinetic-based traffic model.

## Impact

- `skcomplex.core.model` gains `convolve`, `shifted`, and `inverse_helmholtz`, and `inverse_laplacian` is kept as the
  zero-screening case.
- `skcomplex.core.compile` gains the adapter that lowers these operators to SciPy (`scipy.signal`, `scipy.fft`,
  `scipy.ndimage`, `scipy.sparse.linalg`) and to scikit-fem matrices on meshes. No new runtime dependency.
- `skcomplex.models.biology.neuroscience`, `skcomplex.models.earth.oceanography`, and
  `skcomplex.models.engineering.traffic` gain one model each, and the inventory in `organize-model-library` moves them
  from `B:add-nonlocal-operators` to implemented.
