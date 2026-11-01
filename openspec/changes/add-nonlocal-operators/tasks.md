# Tasks

## 1. Inversion

- [x] 1.1 Add `inverse_helmholtz` to `skcomplex.core.laws.utils`, make `inverse_laplacian` its zero-screening case, and
  add the `screening` issue to the pass compiler; verify the negative screening scenario and that the spectral flow
  models still pass
- [x] 1.2 Lower the inversion in `FieldLowering` on spectral fields and on finite-difference lattices with `scipy.fft`
  (`rfftn`, `dstn`, `dctn`) using the eigenvalues of the discrete stencil; verify the single Fourier mode, closed basin,
  and unscreened periodic inversion scenarios
- [x] 1.3 Lower the inversion on networks and finite-element meshes with a cached `scipy.sparse.linalg.factorized`
  factorisation of the graph Laplacian or the scikit-fem matrices; verify a manufactured solution on an L-shaped mesh
  converges at second order and the network solution has zero mean

## 2. Composition

- [x] 2.1 Let `FieldLowering._shift` read the values of a `FieldOperatorKernel` at neighbouring sites; verify the
  velocity from vorticity in a basin scenario
- [x] 2.2 Mark convolutions and inversions in the sparsity pattern of `ProblemSpec.jacobian`; verify an implicit solver
  agrees with an explicit one on a screened diffusion problem within 1e-6

## 3. Convolution

- [x] 3.1 Add `convolve` and `displacement` to `skcomplex.core.laws.utils` with the `kernel-state` issue; verify the
  kernel that reads a state scenario
- [x] 3.2 Sample kernels from the current parameters and lower them with `scipy.fft` on periodic lattices,
  `scipy.signal.oaconvolve` with padding at fixed and zero-flux edges, and `scipy.ndimage.convolve` for small kernels;
  verify the Gaussian smoothing, truncated domain, and wider kernel scenarios
- [x] 3.3 Compose convolutions with the other Fourier multipliers on spectral fields; verify a convolution of a
  Laplacian matches the Laplacian of the convolution within 1e-10
- [x] 3.4 Report convolutions and shifted reads on networks and meshes, spectral operators on non-periodic lattices,
  and operands over several spaces with the codes `law-space` and `operator-spaces`; verify the convolution on a network
  and all misuses at once scenarios

## 4. Shifted reads

- [x] 4.1 Add `shifted` to `skcomplex.core.laws.utils`, lowering whole-site displacements to neighbour tables; verify
  the whole-site shift scenario and that the kernel still compiles to Numba
- [x] 4.2 Add the `displacement` operand of `FieldOperatorKernel` and lower other displacements to
  `scipy.ndimage.map_coordinates` with the boundary mapped to its modes; verify the look-ahead interpolation and
  look-ahead beyond a fixed edge scenarios

## 5. Couplings on fields

- [x] 5.1 Lower a constant times `(other.x - member.x)` in a `CouplingInteraction` on a `FieldElement` to `laplacian`, so
  spectral and finite-element fields keep their own operator, and make couplings on fields read a fixed boundary value
  as `laplacian` does; verify the linear exchange and diffusive coupling scenarios, and agreement with the edge kernel
  away from fixed edges
- [ ] 5.2 Add the optional `boundary=` argument of `inverse_helmholtz`, cubic interpolation (`order=3`) for shifted reads,
  and the sparse path for operands whose states have different boundary conditions, as the design describes; verify
  each against a direct computation

## 6. Models

- [ ] 6.1 Add `amari_neural_field` under `skcomplex.models.biology.neuroscience`; verify the Amari bump scenario
- [ ] 6.2 Add `double_gyre` under `skcomplex.models.earth.oceanography`, streaming the streamfunction through an
  observable of its catalog entry; verify the double gyre circulation scenario
- [ ] 6.3 Add `gkt_traffic` under `skcomplex.models.engineering.traffic`; verify the stop-and-go waves scenario
- [ ] 6.4 Add `active_viscoelastic_gel` under `skcomplex.models.physics.active_matter`; verify the contractile gel
  scenario
- [ ] 6.5 Rebuild the couplings of `legion` and `nanotube_gap_junction_exchange` on `CouplingInteraction`, removing the
  coupling terms from `TermanWangElement` and `ExchangingCellElement`; verify the rebuilt couplings scenario
- [ ] 6.6 Mark the four new models implemented in `docs/overview/inventory.toml` and add them to the gallery; verify the
  inventory check passes

## 7. Documentation and gate

- [ ] 7.1 Document the inversion, kernels and convolution, shifted reads, and couplings on fields in the user guide's
  fields page with runnable examples, and add the changelog entry; verify the docs build runs them
- [ ] 7.2 Run formatting, checks, the docs build, and tests through nox; verify all of them pass
