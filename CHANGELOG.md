# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

### Features

- `StateSpec(tensor=symmetry)` declares rank-2 tensor states, which store their independent components and which laws
  read as Sage matrices, and `TensorSpec` holds those components. `RateLaw` takes a matrix rate, and `FieldElement` takes
  a matrix as a fixed boundary value.
- `PassCompiler` reports a rate that breaks the symmetry of a tensor state with the code `tensor-symmetry`.
- `vector_gradient`, `tensor_divergence`, `symmetric_part`, `antisymmetric_part`, `corotational`, `upper_convected`,
  `leray_projection`, and `distribute_operators` are new law utilities.
- `AlignmentAnalysis` has the kind `'principal-axis'`, `DefectAnalysis` finds point defects by their winding, and
  `AxisVisual` streams axes and arrows to Rerun.
- `dry_active_nematics`, `active_gels`, and `LandauDeGennesEnergyInteraction` join `physics/active_matter`.
- `PhaseSpace(position, axes)` and `AxisSpec` hold fields over position and class axes, `coordinate` reads their
  coordinates in laws, and `moment` takes `axes`.
- `EncounterInteraction` runs at every position of a phase space and takes no `field` for a linear redistribution.
- `CollisionInteraction` and `CollisionLaw` give the binary collision integral by the fast spectral method, and
  `maximum_entropy` gives the equilibrium with the moments of a density.
- `ConservationInteraction` and `AdvectionInteraction` take `sources`, and `FluxLaw` fluxes read moments and field
  operators.
- `SemiLagrangianSolver` and `StrangSplittingSolver` advance transport and collisions, and `ProblemSpec.sources` keeps
  the source rates apart from the flux differences.
- `ImageVisual` takes `slices` and `moments`.
- The library gains `boltzmann_equation`, `bgk_equation`, `vlasov_equation`, `enskog`, and `bertin_droz_gregoire`.

## [0.1.0](https://github.com/georgedouzas/sage-physics/releases/tag/0.1.0) - 2023-09-19

<small>[Compare with first commit](https://github.com/georgedouzas/sage-physics/compare/eb1319601a393032e7230466e82e3cebb325f57c...0.1.0)</small>

### Bug Fixes

- Use correct type definition ([23120a1](https://github.com/georgedouzas/sage-physics/commit/23120a151149dbb05f452918ecad1b13cfdfe0c0) by georgedouzas).

### Code Refactoring

- Prepare stable release ([ceadea6](https://github.com/georgedouzas/sage-physics/commit/ceadea6673406bbdae813778f1dd06f242cb5858) by georgedouzas).

### Chore

- Rename to main ([5b9b89d](https://github.com/georgedouzas/sage-physics/commit/5b9b89d918e5afe92a07ea4da337bf3213f79678) by georgedouzas).
- Remove cli files ([9ccc379](https://github.com/georgedouzas/sage-physics/commit/9ccc3799280529a616f233196da72567e2218f85) by georgedouzas).

