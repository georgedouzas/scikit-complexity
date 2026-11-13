# Design

## Context

`src/skcomplex/core/analysis/_operator.py` defines `operator_measures`, its result, and five private helpers. It
reaches past the public surface of the core in three ways, which the move has to resolve, since the import contract
"Systems import only public modules" forbids `skcomplex.systems` to import private modules of the core:

- It compiles an operator system with `OperatorCompiler` from the private module
  `engine/compile/passes/_operator_compiler.py`, reads `DENSE_LIMIT` from the same module, and checks the result
  against `OperatorProblemSpec` from the private module `engine/problems/_operator_problem_spec.py`.
- It calls the private helpers `_check_choice`, `_check_settings`, `_composite`, and `_read_system` of
  `core/analysis/_base.py`.
- Its helper `_operator_problem` is also imported by `core/analysis/_symbolic.py`, which returns the operator
  Hamiltonian for the kind `'hamiltonian'` of `symbolic`.

`core/analysis/_discovery.py` imports `operator_measures` and lists it among the analyses that `applicable_analyses`
binds. The core cannot import `skcomplex.systems`, so the moved analysis cannot stay on that list.

The metabolism analyses in `systems/biology/common/metabolism/_analyses.py` are the precedent for a field analysis.
They import only public core modules, such as `skcomplex.core.analysis.types`, `skcomplex.core.engine.problems`, and
`skcomplex.core.engine.runtime`, keep their own private `_check_choice`, and are absent from discovery.

## Goals / Non-Goals

**Goals:**

- `operator_measures` behaves exactly as now from its new place: the same kinds, settings, errors, and results.
- The moved module imports only public modules of the core, so the import contracts hold unchanged.
- `symbolic(..., kind='hamiltonian')` keeps working with no import from the systems.

**Non-Goals:**

- Moving the operator solvers, the operator compiler, `SectorSpec`, `OperatorSpec`, or `local_operator`. The engine
  handles operator systems by type, so by the placement rule they stay in the core.
- Renaming `operator_measures` or changing its settings.
- Reviewing other analyses. The review behind the proposal kept every other module of `skcomplex.core.analysis`.
- Letting the discovery of the core find field analyses. That would need a registry the core reads, which is a change
  of its own.

## Decisions

### The analysis goes to `_measures.py` of the quantum common package

The new module is `src/skcomplex/systems/physics/common/quantum/_measures.py`, exported from the package's
`__init__.py` with `operator_measures` and `OperatorMeasuresResult`. The quantum package already holds the qubit and
the bosonic mode, which are what the analysis measures, and it serves the quantum systems of
`physics/statistical_mechanics` and `physics/waves`. Its tests go to
`tests/systems/physics/common/quantum/test_measures.py`, the mirrored path.

Alternative: a new family package such as `systems/physics/common/quantum_analysis`. Rejected, since the family is the
same, and a new package needs the maintainer's approval for no gain.

### The operator compiler and problem become public engine names

`OperatorCompiler` joins `__all__` of `skcomplex.core.engine.compile` and `OperatorProblemSpec` joins `__all__` of
`skcomplex.core.engine.problems`. The moved analysis imports them from these packages, which the import contract
allows. Both are engine-typed primitives, which the placement rule keeps in the core, so making them public changes
no placement. Their docstrings get the examples a public class needs. The requirement that compilers and problem forms
are private names these two as its exceptions, and the tests of the public surface of both packages list them.

Alternative: compile through `PassCompiler` and check `form == FormKind.OPERATOR`. Rejected, since the analysis needs
the fields of `OperatorProblemSpec`, such as its generator and its subsystem dimensions, and a public type is what
mypy and readers need to see them.

### Guide pages may import a field analysis

The documentation rule lets guide pages import only system classes from subfield packages. The quantum guide page
imports `operator_measures` from the family package `systems/physics/common/quantum`, so the rule and its convention
check accept an analysis function or its result from a family package of a field, and the prose may name that
package. Showing the analysis only in the gallery was the alternative, rejected because measuring a quantum system is
what the page teaches.

### The analysis keeps its own private checks

The moved module writes its own private `_check_choice`, `_check_settings`, and source reading, as the metabolism
analyses do, instead of importing the private helpers of `core/analysis/_base.py`. Reading a source here is short:
a trajectory gives its problem, a ready-made system gives its prepared composite, and a composite or a problem is
used as it is. The messages stay word for word, so the tests keep their `match=` strings.

Alternative: make the helpers of `_base.py` public. Rejected, since they are plumbing of the core analyses, not
operations a user calls, and publishing them would widen the public surface for one caller.

### The sparse threshold is a constant of the analysis

`DENSE_LIMIT` of the operator compiler decides when the analysis switches to a sparse eigensolver. The moved module
defines its own public constant with the same value, `DENSE_LIMIT = 1024`, at the top of the module, since the
threshold belongs to the diagonalisation, not to compilation. The compiler keeps its constant for its own use.

### `_operator_problem` stays in the core inside `_symbolic.py`

The function moves into `core/analysis/_symbolic.py` as a private function, its only remaining user in the core. The
moved analysis has its own equivalent built on the public `OperatorCompiler` and `OperatorProblemSpec`. The two copies
are four lines each and serve different layers, so the duplication is accepted rather than adding a public helper.

### Discovery drops the analysis

`_discovery.py` removes `operator_measures` from its imports and its list. A quantum trajectory still gets every core
analysis that applies to its series.

### No alias and no history test

`skcomplex.core.analysis` drops both names with no alias, as for earlier breaking moves. No test pins the absence of
the old names. The tests check the behaviour of the analysis at its new place.

## Risks / Trade-offs

- [Users who import `operator_measures` from `skcomplex.core.analysis` get an `ImportError`] → The release notes list
  the move with the new import line, and the quantum guide page shows it.
- [The two copies of the operator problem helper drift apart] → Both are short and check one type, and each has
  tests through its own analysis: `symbolic(..., kind='hamiltonian')` and `operator_measures`.
- [The two `DENSE_LIMIT` constants drift apart] → They answer different questions, the size up to which the compiler
  builds dense matrices and the size up to which the analysis diagonalises densely, so they may differ in the future
  without harm.

## Migration Plan

Replace `from skcomplex.core.analysis import operator_measures` with
`from skcomplex.systems.physics.common.quantum import operator_measures`. Nothing else changes for the caller.
