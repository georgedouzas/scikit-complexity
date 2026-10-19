# Tasks

## 1. Inversion

- [ ] 1.1 Add `inverse_helmholtz` to `skcomplex.core.model` with the screening check, and make `inverse_laplacian` its
  zero-screening case, and verify the negative screening scenario and that the existing spectral flow models still
  pass
- [ ] 1.2 Lower the inversion on spectral fields and on finite-difference lattices with `scipy.fft` (`rfftn`, `dstn`,
  `dctn`) using the eigenvalues of the discrete stencil, and verify the single Fourier mode, closed basin, and
  unscreened periodic inversion scenarios
- [ ] 1.3 Lower the inversion on networks and finite-element meshes with a cached `scipy.sparse.linalg.factorized`
  factorisation of the graph Laplacian or the scikit-fem matrices, and verify a manufactured solution on an L-shaped
  mesh converges at second order and the network solution has zero mean
- [ ] 1.4 Document the inversion in the user guide's fields page with a runnable example, and verify the docs build
  runs it

## 2. Composition

- [ ] 2.1 Let `FieldLowering.shift` read derived values at neighbouring sites, and verify the velocity from vorticity
  in a basin scenario
- [ ] 2.2 Mark convolutions and inversions in the Jacobian sparsity pattern, and verify an implicit stepper agrees with
  an explicit one on a screened diffusion problem within 1e-6

## 3. Convolution

- [ ] 3.1 Add `convolve` and the displacement symbols to `skcomplex.core.model`, with the check that kernels read no
  state, and verify the kernel that reads a state scenario
- [ ] 3.2 Sample kernels from the current parameters and lower them with `scipy.fft` on periodic lattices,
  `scipy.signal.oaconvolve` with padding at fixed and zero-flux edges, and `scipy.ndimage.convolve` for small kernels,
  and verify the Gaussian smoothing, truncated domain, and wider kernel scenarios
- [ ] 3.3 Compose convolutions with other spectral multipliers on spectral fields, and verify a convolution of a
  Laplacian matches the Laplacian of the convolution within 1e-10
- [ ] 3.4 Reject convolutions on networks and meshes, and verify the convolution on a network scenario
- [ ] 3.5 Document kernels and convolution in the user guide with a runnable example, and verify the docs build runs it

## 4. Shifted reads

- [ ] 4.1 Add `shifted`, lowering whole-site displacements to neighbour tables, and verify the whole-site shift
  scenario and that the kernel still compiles to Numba
- [ ] 4.2 Lower other displacements to `scipy.ndimage.map_coordinates` with the boundary mapped to its modes, and
  verify the look-ahead interpolation and look-ahead beyond a fixed edge scenarios
- [ ] 4.3 Document shifted and look-ahead reads in the user guide with a runnable example, and verify the docs build
  runs it

## 5. Library

- [ ] 5.1 Add the Amari neural field under `skcomplex.models.biology.neuroscience`, registered at
  `biology/neuroscience` with its source, and verify the Amari bump scenario and the conformance suite
- [ ] 5.2 Add the quasi-geostrophic double gyre under `skcomplex.models.earth.oceanography`, registered at
  `earth/oceanography` with its source, streaming the streamfunction as an observable image, and verify the double
  gyre circulation scenario and the conformance suite
- [ ] 5.3 Add the gas-kinetic-based traffic model under `skcomplex.models.engineering.traffic`, registered at
  `engineering/traffic` with its source, and verify the stop-and-go waves scenario and the conformance suite
- [ ] 5.4 Mark the three models as implemented in the inventory of the model library, add them to the gallery, and
  verify the layout test and the inventory check pass

## 6. Gate

- [ ] 6.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
