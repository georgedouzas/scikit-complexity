# Tasks

## 1. Phase spaces and coordinates

- [ ] 1.1 Add `AxisSpec` to `skcomplex.specs` and `PhaseSpace` to `skcomplex.core.spaces`, a lattice space with a
  boundary and coordinates per class axis that `FieldElement` accepts, with docstrings and examples, and the
  `space-size` issue of the compile report, and verify the size of a phase space scenario and the size limit issue
- [ ] 1.2 Add `coordinate` to `skcomplex.core.laws.utils` and its lowering to broadcast coordinate arrays on NumPy and
  Numba, with the `law-space` issue for a coordinate outside a phase space, and verify the Maxwellian start scenario

## 2. Transport and moments

- [ ] 2.1 Lower drifts that read coordinates in the finite-difference, finite-volume, and spectral discretizations of
  `AdvectionInteraction` on a `PhaseSpace`, and verify the free streaming, zero drift, and periodic class axis
  scenarios
- [ ] 2.2 Add `axes` to `moment`, reducing over class axes into a field over position, keep its behaviour without
  `axes`, and verify the shifted Maxwellian scenario and the existing kinetic model tests
- [ ] 2.3 Add `sources` roles to `ConservationInteraction` and `AdvectionInteraction`, lower operators on moments along
  the position axes, and verify the Landau damping scenario

## 3. Encounters, equilibria, and collisions

- [ ] 3.1 Batch the encounter, destruction, and proliferation kernels over positions on a `PhaseSpace`, make the
  `field` of `EncounterInteraction` an optional role with the linear redistribution, and verify the linear
  redistribution scenario and the existing encounter tests
- [ ] 3.2 Add `maximum_entropy` to `skcomplex.core.laws.utils` with its Newton lowering per position, and verify the
  conservation under BGK scenario
- [ ] 3.3 Add `CollisionLaw` with its `RequirementSpec` to `skcomplex.core.laws` and `CollisionInteraction` to
  `skcomplex.core.interactions`, with the fast spectral weights on `scipy.fft`, the density factor, and the `law-space`
  issue, and verify the BKW, density factor, and collision without velocity axes scenarios and agreement with direct
  quadrature on a coarse grid

## 4. Solvers and visualization

- [ ] 4.1 Record the linear drift of every axis on the problem, and add `SemiLagrangianSolver` with its `SupportSpec`
  on `scipy.fft` shifts and `scipy.ndimage.map_coordinates`, and verify the large transport steps and nonlinear flux
  scenarios
- [ ] 4.2 Keep the source rates of `ProblemSpec` apart from the flux differences, add `StrangSplittingSolver` with its `SupportSpec`, and verify the second order splitting scenario
- [ ] 4.3 Add the `slice` and `moments` arguments of `ImageVisual`, and verify the two-stream viewer scenario with a
  recording in memory

## 5. Library and documentation

- [ ] 5.1 Add `BhatnagarGrossKrookInteraction` and the Boltzmann, BGK, Vlasov-Poisson, and local Enskog factories to
  `physics/fluids`, and verify the Sod shock tube and Landau damping model scenarios and the conformance suite
- [ ] 5.2 Add `AngularAlignmentInteraction`, `AngularDiffusionInteraction`, and the Bertin-Droz-Grégoire factory to
  `physics/active_matter`, and verify the flocking onset scenario and the conformance suite
- [ ] 5.3 Move `boltzmann_equation`, `bgk_equation`, `vlasov_equation`, `enskog`, and `bertin_droz_gregoire` from
  `blocked add-phase-space-fields` to implemented in `docs/overview/inventory.toml`, and verify the inventory check
  passes
- [ ] 5.4 Document phase spaces, coordinates, moments, encounters, equilibria, collisions, mean fields, and the solvers
  in the user guide, with the library survey's outcome, add a changelog entry, and verify the documentation build runs
  the examples

## 6. Gate

- [ ] 6.1 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
