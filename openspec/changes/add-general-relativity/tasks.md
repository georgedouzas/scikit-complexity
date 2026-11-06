# Tasks

## 1. Dependency

- [x] 1.1 Check that EinsteinPy 0.5.0 installs and computes the Schwarzschild Einstein tensor on Python 3.12, 3.13, and
  3.14 with NumPy 2.5.4 and SymPy 1.14.0
- [x] 1.2 Check that Sage expressions convert to SymPy and back through EinsteinPy, functions of an ansatz included
- [x] 1.3 Add `einsteinpy>=0.5.0` as a runtime dependency and update `pdm.lock`

## 2. Core

- [x] 2.1 Add `PseudoRiemannianSpace` with the Christoffel symbols and, optionally, the curvature tensors
- [x] 2.2 Add `GeodesicInteraction`, the geodesic equation as a rate law of a vector position of order 2
- [x] 2.3 Add `EinsteinInteraction`, the mixed Einstein equations of an ansatz solved for the second time derivatives,
  with lattice axes for `1 + 1` fields
- [x] 2.4 Test the space, the geodesics, and the Einstein equations, their errors included

## 3. Systems

- [x] 3.1 Add `SchwarzschildSpace`, `KerrSpace`, and `GeodesicParticleElement` to `physics/celestial_mechanics`
- [x] 3.2 Add `schwarzschild_precession`, `light_bending`, and `kerr_frame_dragging` with tests against the precession,
  the exact deflection, and the angular velocity of the frames
- [x] 3.3 Add `RobertsonWalkerSpace`, `LemaitreTolmanBondiSpace`, `HomogeneousUniverseElement`, and `DustShellElement`
  to `physics/cosmology`
- [x] 3.4 Add `flrw_cosmology` and `oppenheimer_snyder_collapse` with tests against the analytic Friedmann solutions and
  the Oppenheimer-Snyder cycloid

## 4. Documentation and checks

- [x] 4.1 Add the primitive `curved-spacetimes` and the inventory entries of the new systems
- [x] 4.2 Add a gallery example of every new system and run each
- [x] 4.3 Run the inventory check, the strict gallery coverage, the catalog tests, and the tests of the subfields
- [x] 4.4 Run ruff, black, docformatter, mypy, and pydoclint on the changed files

## 5. Review follow-ups

- [x] 5.1 Derive the tensors of a space again only when its metric, coordinates, or curvature flag change by value
- [x] 5.2 Document the space and the two interactions in the user guide, with blocks checked by `--markdown-docs`
