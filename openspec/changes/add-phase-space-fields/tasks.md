# Tasks

## 1. Phase spaces and coordinates

- [ ] 1.1 Add `PhaseSpace`, `VelocityAxis`, and `OrientationAxis`, accepted by `Field`, with docstrings and examples,
  and verify the size and initial-values scenario and the size limit error
- [ ] 1.2 Add phase coordinate symbols and their lowering to broadcast coordinate arrays on NumPy and Numba, and
  verify the Maxwellian start scenario

## 2. Transport and moments

- [ ] 2.1 Add `streaming` and `acceleration` as expansions into `advection`, with array speeds in the upwind,
  finite-volume, and spectral lowering, and verify the free streaming, zero force, and periodic orientation scenarios
- [ ] 2.2 Add `moment` as a reduction over velocity and orientation axes in laws and as an observable, sharing the
  quadrature helper with `add-kinetic-models` if that change has landed, and verify the shifted Maxwellian scenario
- [ ] 2.3 Add mean-field reads between phase-space and position fields, and verify the Landau damping scenario

## 3. Collision operators

- [ ] 3.1 Add `BGK` with the conservative discrete Maxwellian, and verify the BGK conservation and entropy scenario
- [ ] 3.2 Add `Collision` by the fast spectral method on `scipy.fft`, with the Enskog factor, and verify the BKW and
  Enskog factor scenarios and agreement with direct quadrature on a coarse grid
- [ ] 3.3 Add `AngularCollision` with self-diffusion and alignment on `scipy.fft`, and verify the self-diffusion
  scenario

## 4. Steppers and visualization

- [ ] 4.1 Add `SemiLagrangian` on `scipy.fft` shifts and `scipy.ndimage.map_coordinates`, registered by name, and
  verify the large Courant number scenario
- [ ] 4.2 Add `StrangSplitting`, registered by name, and verify the second-order splitting scenario
- [ ] 4.3 Add the `slice` argument of `ImageVisual` and series for one-dimensional moments, and verify the two-stream
  viewer scenario with a recorded Rerun stream

## 5. Library and documentation

- [ ] 5.1 Add the Boltzmann, BGK, Vlasov-Poisson, and local Enskog factories under `physics/fluids`, and verify the
  Sod shock tube and Landau damping model scenarios and the conformance suite
- [ ] 5.2 Add the Bertin-Droz-Grégoire factory under `physics/active_matter`, and verify the flocking onset scenario
  and the conformance suite
- [ ] 5.3 Move the five models from `blocked add-phase-space-fields` to implemented in the inventory, and verify the
  inventory check passes
- [ ] 5.4 Document phase spaces, transport, moments, collisions, mean fields, and the steppers in the user guide, with
  the library survey's outcome, add a changelog entry, and verify the documentation test passes

## 6. Gate

- [ ] 6.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
