# Tasks

## 1. Tensor states

- [x] 1.1 Add `tensor` to `StateSpec` with its check against `vector`, the expansion into independent components in
  `EntrySpec` with the kind `DomainKind.TENSOR`, and the matrix view in the namespace, and verify the two- and
  three-dimensional symmetric traceless and the vector and tensor together scenarios in
  `tests/specs/test_state_spec.py` and `tests/specs/test_entry_spec.py`
- [x] 1.2 Read the rates of a `RateLaw` on a tensor state from the entries of its matrix value, and verify that a
  relaxation `RateLaw(s.Q, -s.Q)` runs on the NumPy and Numba backends
- [ ] 1.3 Accept a matrix as the fixed boundary value of a tensor state in `FieldElement`, and verify the anchoring at
  walls scenario and the rejected asymmetric matrix
- [x] 1.4 Add the issue `tensor-symmetry` to `PassCompiler` with its numeric fallback, and verify the law that breaks
  the symmetry scenario in `tests/engine/compile/test_pass_compiler.py`

## 2. Tensor operations and derivatives

- [x] 2.1 Add `vector_gradient`, `tensor_divergence`, `symmetric_part`, and `antisymmetric_part` in a module `_tensors`
  of `skcomplex.core.laws.utils`, and verify the divergence of a tensor, strain rate and vorticity, and tensor
  divergence of a scalar scenarios
- [x] 2.2 Add `corotational` and `upper_convected`, and verify the upper-convected identity and the rotation by a
  uniform vorticity scenarios
- [x] 2.3 Add `leray_projection`, and verify the divergence-free projection and the Stokes flow between walls scenarios
- [x] 2.4 Document tensor states, the operations, the derivatives, and the projection in the user guide's fields page
  with a runnable example, and verify the documentation test passes

## 3. Analysis and visualization

- [x] 3.1 Add the kind `'principal-axis'` to `AlignmentAnalysis`, and verify the axes and scalar order of a uniform
  tensor field
- [x] 3.2 Add `DefectAnalysis` to `skcomplex.analysis`, and verify the defect pair and polar vortex scenarios
- [x] 3.3 Add `AxisVisual` to `skcomplex.engine.viz`, and verify the nematic in the viewer scenario with a recorded
  Rerun stream

## 4. Library

- [x] 4.1 Add `LandauDeGennesEnergyInteraction` and `dry_active_nematics` to `skcomplex.models.physics.active_matter`,
  and verify the density conservation scenario and the conformance suite
- [ ] 4.2 Add `active_gels` with the nematic and polar orders, and verify the passive nematic order, active turbulence,
  and spontaneous flow scenarios and the conformance suite
- [ ] 4.3 Mark `dry_active_nematics` and `active_gels` implemented in `docs/overview/inventory.toml`, add their gallery
  examples under `docs/examples/physics/active_matter`, and verify the inventory check and the gallery coverage check
- [ ] 4.4 Add the changelog entry for tensor states and the new models, and verify the changelog session passes

## 5. Gate

- [ ] 5.1 Run `nox -s formatting`, `nox -s checks`, `nox -s docs`, and `nox -s tests` in that order, and verify all of
  them pass
