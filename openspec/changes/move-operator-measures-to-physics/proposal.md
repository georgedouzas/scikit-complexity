# Proposal

## Why

`skcomplex.core.analysis` is meant to hold general measurements, ones that apply to the runs of systems from any
field. A review of its modules found that almost every analysis meets this test. Some carry the name of one field but
compute a general operation: `alignment`, `defects`, `autocatalytic_sets`, `elementary_modes`, and `chain_shape` each
serve systems of two fields or more. One does not. `operator_measures` computes the spectrum, ground state, purity, and
entanglement of a quantum operator system. Purity, entanglement, and the Hilbert space sectors it reads exist only in
quantum physics, so it is a measurement of one field that sits in the general core.

The placement rule already sends a component that one field uses to `systems/<field>/common/<family>`. It says nothing
about analyses. The metabolism analyses of `skcomplex.systems.biology.common.metabolism`, such as `flux_variability`,
already follow the rule for components, so the practice exists but is not written down. The maintainer chose to write
it down for analyses too: a field analysis lives in its field's common package.

## What Changes

- **BREAKING** `operator_measures` and `OperatorMeasuresResult` move from `skcomplex.core.analysis` to
  `skcomplex.systems.physics.common.quantum`, beside `QubitElement` and `ModeElement`. The name and the settings stay.
  No alias is kept in `skcomplex.core.analysis`.
- The placement rule covers analyses: an analysis that measures what several fields share stays in
  `skcomplex.core.analysis`, and an analysis of one field goes to `systems/<field>/common/<family>`.
- `applicable_analyses` lists the analyses of the core only, so `operator_measures` leaves its list, as the metabolism
  analyses are already absent from it.
- `OperatorCompiler` becomes a public name of `skcomplex.core.engine.compile`, and `OperatorProblemSpec` a public name
  of `skcomplex.core.engine.problems`, since the moved analysis compiles an operator system and systems import only the
  public modules of the engine. Both are engine-typed primitives, which the placement rule keeps in the core.
- The helper `_operator_problem`, which the symbolic analysis also reads for its `'hamiltonian'` kind, stays in the
  core as a private function of the symbolic analysis module.
- The tests of `operator_measures` move to the mirrored path of the new module, and the guide page on quantum systems,
  the two gallery examples, and the tests of the quantum systems import it from the quantum package.
- The three texts of the placement rule, in `openspec/constitution.md`, `CONTRIBUTING.md`, and the extension page of
  the user guide, state that the rule covers analyses.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `package-layout`: the placement rule covers analyses as well as components.
  The operator compiler and the operator problem become public exceptions to the private compilers and problem forms.
- `user-documentation`: guide pages may import an analysis of one field and its result from the family package of the
  field.
- `quantum-systems`: the spectra, ground states, purity, and entanglement come from the quantum package of physics,
  not from the general analysis layer.
- `analysis-interface`: the core analyses are those that measure what several fields share, and discovery lists only
  them.

## Impact

- Public API: `skcomplex.core.analysis.operator_measures` and `OperatorMeasuresResult` are removed, and
  `skcomplex.systems.physics.common.quantum` gains them. `skcomplex.core.engine.compile.OperatorCompiler` and
  `skcomplex.core.engine.problems.OperatorProblemSpec` become public. The next release notes list the move.
- Code: `src/skcomplex/core/analysis/_operator.py` is removed, `_symbolic.py`, `_discovery.py`, and the analysis
  `__init__.py` change, and `src/skcomplex/systems/physics/common/quantum/_measures.py` is added.
- Tests: `tests/core/analysis/test_operator.py` becomes
  `tests/systems/physics/common/quantum/test_measures.py`, and two system tests change their import.
- Docs: `docs/overview/user_guide/build_a_custom_system/quantum_systems.md` and two gallery examples change their
  import, and the constitution, `CONTRIBUTING.md`, and `docs/overview/user_guide/extend_the_library.md` extend the
  placement rule.
- Dependencies: none. QuTiP stays a runtime dependency, since the operator solvers of the engine use it.
