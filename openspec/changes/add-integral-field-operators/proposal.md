# Proposal

## Why

Three models of the reference audit cannot be written today. A density over group composition whose groups meet in
pairwise conflicts integrates against a victory probability `rho(z, u)` that is no function of `z - u`, the light that
reaches a tree integrates the canopy above it from its height to the top of the domain, and a density over three
strategy frequencies lives on the 2-simplex rather than on a box. The core convolves fields with kernels of the
displacement and reduces them over whole class axes, but it has no two-point kernel, no cumulative integral, and no
domain other than a box.

## What Changes

- Add a two-point integral operator to the law utilities, `integral_transform(expression, kernel, axes)`, the integral
  over named class axes of a phase space of a kernel `K(c, c')` times an expression at `c'`. The kernel reads the
  coordinate of the point a law updates, `coordinate(name)`, the coordinate of the point integrated over,
  `integration_variable(name)`, and parameters.
- Add a cumulative integral to the law utilities, `cumulative_integral(expression, axis, reverse=False)`, the integral
  of an expression along one axis of a lattice from the start of the domain to every site, or from every site to the end.
- Make `convolve` act along the position axes of a phase space, so a density over position and class disperses over
  position at every class.
- Add `SimplexSpace`, a phase space whose class axes are the free frequencies of a simplex, cut into whole cells and
  partial cells along its slanted face. Finite-volume fluxes pass only through the open part of every face and are
  divided by the volume of the cell, and the reductions and integrals over class axes weight every cell by its volume.
- Add the catalog systems `nonlinear_pairwise_group_conflict` and `trimorphic_multilevel_selection` under
  `social/cooperation` (Cooney 2024, arXiv 2405.18419 §6 and §7) and `forest_size_structure` under `biology/ecology`
  (Grilletta et al. 2026, arXiv 2608.11918 Eq. 10), with tests, inventory entries, and gallery examples.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `nonlocal-operators`: adds two-point integral transforms, cumulative integrals, and convolutions along the position
  axes of phase spaces.
- `phase-space-fields`: adds simplex domains of class axes with their finite-volume fluxes and weighted reductions.
- `model-library`: adds the three systems.

## Impact

- `skcomplex.core.components.laws.utils`: new `integral_transform`, `integration_variable`, and `cumulative_integral`.
- `skcomplex.core.components.spaces`: new `SimplexSpace` in a new module of the existing package.
- `skcomplex.core.engine.compile`: lowering of the new operators, convolution on phase spaces, the compile report
  issues, and the weighted reductions.
- `skcomplex.core.engine.problems`: the finite-volume kernel takes cell volumes and face apertures.
- `skcomplex.systems.social.cooperation` and `skcomplex.systems.biology.ecology`: three new systems.
- `docs`: user guide sections, inventory entries, and gallery examples. No new dependency.
