# Tasks

## 1. Dependency and adapter

- [ ] 1.1 Add `py-pde` 0.58 or later to the runtime dependencies in `pyproject.toml` and the Project Profile of the
  constitution, and verify `pdm install`, `deptry`, and `pip-audit` pass
- [ ] 1.2 Add the private adapter module of the NumPy and Numba backends in `skcomplex.core.backends` that builds a `py-pde` `CartesianGrid` from a lattice,
  translates periodic, zero-flux, and fixed tensor boundaries, and wraps compiled operators and Poisson solves as
  derived reads, and verify its tests against `py-pde` on a scalar Laplacian and a Poisson problem with a known solution

## 2. Tensor states

- [ ] 2.1 Give the existing `Tensor` domain its expansion into independent components and the matrix and vector views
  in the symbols namespace, and verify the two- and three-dimensional symmetric traceless scenarios
- [ ] 2.2 Add the validation issue `tensor-symmetry` for rate laws on tensor states with its numeric fallback, and the
  check of tensor states outside the sites of a lattice, and verify the law that breaks the symmetry and tensor state on
  a network scenarios
- [ ] 2.3 Add `DomainKind.TENSOR` to the declared support of the ODE and stiff solvers, the spectral discretization,
  and the NumPy and Numba backends, and the requirements of the tensor operators, and verify the solver without tensor
  support and tensor operator on a scalar scenarios and the declaration test
- [ ] 2.4 Add fixed tensor boundary values, and verify the anchoring scenario

## 3. Tensor calculus and objective derivatives

- [ ] 3.1 Add `vector_gradient`, `tensor_divergence`, `strain_rate`, and `vorticity_tensor` to `skcomplex.core.laws` with their lowering to
  stencils, spectral multipliers, and the adapter, and verify the divergence of a tensor and the strain and vorticity
  scenarios on NumPy and Numba
- [ ] 3.2 Add `corotational` and `upper_convected`, and verify the upper-convected identity and the uniform vorticity
  rotation scenarios
- [ ] 3.3 Document tensor states, the operators, and the objective derivatives in the user guide's fields page with a
  runnable example, and verify the documentation test passes

## 4. Incompressible velocities

- [ ] 4.1 Add `stokes_velocity` and `project` on spectral fields in two and three dimensions, and verify the
  divergence-free Stokes velocity scenario
- [ ] 4.2 Add the bounded two-dimensional Stokes solve through the adapter, and verify the Poiseuille scenario
- [ ] 4.3 Add the elastic diffusivity of tensor laws to the explicit stability check, and verify the warning names the
  limit for a nematic relaxation run with too large a step
- [ ] 4.4 Document incompressible velocities from forces in the user guide, and verify the documentation test passes

## 5. Analysis and visualization

- [ ] 5.1 Add `principal_axis` and `axis_defects` to `skcomplex.core.analysis`, and verify the defect pair scenario and
  a polar `+1` vortex
- [ ] 5.2 Add `AxisVisual` under the kind `'axis'`, and verify the nematic viewer scenario with a recorded
  Rerun stream
- [ ] 5.3 Document the analysis functions and the axis visual with runnable examples, and verify the documentation
  test passes

## 6. Library

- [ ] 6.1 Add the dry active nematic to `physics/active_matter`, and verify the density conservation scenario and the
  conformance suite
- [ ] 6.2 Add the active nematic gel, and verify the passive nematic order and active turbulence scenarios and the
  conformance suite
- [ ] 6.3 Add the active polar gel, and verify the spontaneous flow scenario and the conformance suite
- [ ] 6.4 Move the three models from blocked to implemented in the inventory, add them to the field page of
  `physics/active_matter`, and verify the inventory check and the layout test pass
- [ ] 6.5 Add the changelog entry for tensor fields and the new models, and verify the changelog session passes

## 7. Gate

- [ ] 7.1 Run formatting, checks, the docs build, and tests through nox in that order, and verify all of them pass
