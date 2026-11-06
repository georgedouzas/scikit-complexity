# Tasks

## 1. Integral operators

- [x] 1.1 Add `integration_variable`, `integral_transform`, and `cumulative_integral` to
  `skcomplex.core.components.laws.utils` in a new module `_integrals`, with docstrings and examples, and verify their
  doctests and the argument errors in `tests/core/components/laws/utils/test_integrals.py`
- [x] 1.2 Lower `integral_transform` over the class axes of a phase space and `cumulative_integral` along a lattice axis
  in `FieldLowering`, sharing the kernel sampling with `convolve`, and verify the kernel of the coordinates, the
  antisymmetric kernel, the tail integral, and the Beer-Lambert scenarios in `test_integrals.py`
- [x] 1.3 Make `convolve` act along the position axes of a phase space, by a window over the positions and by the FFT
  along them for periodic positions, and verify the dispersal scenario against direct one-dimensional convolutions in
  `tests/core/engine/compile/test_field_lowering.py`
- [x] 1.4 Add the `kernel-state` issue of two-point kernels and the `law-space` issues of an integral over class axes
  without a phase space and of a cumulative integral off a lattice, and verify them in
  `tests/core/engine/compile/test_pass_compiler.py`

## 2. Simplex domains

- [x] 2.1 Add `SimplexSpace` to `skcomplex.core.components.spaces` with `volumes_` and `apertures_` from
  `scipy.stats.irwinhall`, raise the `scipy` lower bound to 1.15 and relock, and verify the volumes of the 2-simplex
  and the 3-simplex, the closed faces, and the axis errors in `tests/core/components/spaces/test_simplex.py`
- [x] 2.2 Pass the volumes and apertures to `FiniteVolumeKernel`, scale its interface fluxes and cell rates, divide its
  largest wave speed by the smallest volume, weight `reduce_axes` by the volumes, and verify the replicator flow and
  moment scenarios in `test_simplex.py` and the wave speed in `tests/core/engine/problems/test_finite_volume_kernel.py`
- [x] 2.3 Reject local operators on a simplex in the lowering and the compile report and partial cells in
  `SemiLagrangianSolver`, and verify the `law-space` issue in `test_pass_compiler.py` and the solver error in
  `tests/core/engine/solvers/test_semi_lagrangian.py`
- [x] 2.4 Document the integrals over class axes and simplex domains in the phase-space fields guide and the cumulative
  integral in the fields guide, and verify their code blocks with `pytest --markdown-docs`

## 3. Systems

- [x] 3.1 Add `nonlinear_pairwise_group_conflict` and `trimorphic_multilevel_selection` to `social/cooperation`, with
  tests of the conserved total, the effect of conflict strength, the fall of defection, and the victory errors in
  `tests/systems/social/cooperation/test_multilevel_selection.py`
- [x] 3.2 Add `forest_size_structure` to `biology/ecology`, with tests of the edge depletion of small trees and of the
  stationary density in `tests/systems/biology/ecology/test_forest_size_structure.py`
- [x] 3.3 Add the primitives `integral-operators` and `simplex-domains` and the three entries to
  `docs/overview/inventory.toml`, and verify the inventory check passes
- [x] 3.4 Add the gallery examples of the three systems, and verify they run and that `gallery_tools coverage --strict`
  passes

## 4. Gate

- [x] 4.1 Run ruff, the formatting checks, mypy and pydoclint on the changed source files, the doctests of the changed
  modules, the catalog tests, and the tests of the touched modules, and verify all of them pass
