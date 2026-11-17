# Tasks

Every test run below sets `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and uses
`.venv/bin/python -m pytest -n 4 -p no:cacheprovider`. Ruff and mypy run from `.nox/checks-3-12-file-src/bin/`, the
import contracts with `PYTHONPATH=src .nox/checks-3-12-file-src/bin/lint-imports`, and the conventions with
`.venv/bin/python openspec/scripts/check_conventions.py`. No test pins the absence of a moved name.

## 1. Public engine names

- [x] 1.1 Export `OperatorCompiler` from `skcomplex.core.engine.compile` and `OperatorProblemSpec` from
  `skcomplex.core.engine.problems`, with a doctest example in each docstring if it has none, verified by ruff, mypy on
  `src`, lint-imports, and the doctests of `engine/compile/passes/_operator_compiler.py` and
  `engine/problems/_operator_problem_spec.py`

## 2. The core keeps the symbolic Hamiltonian

- [x] 2.1 Move `_operator_problem` from `core/analysis/_operator.py` into `core/analysis/_symbolic.py` as a private
  function, and import it there from nowhere else, verified by mypy on `src` and the tests of
  `tests/core/analysis/test_symbolic.py`

## 3. The analysis moves to the quantum package

- [x] 3.1 Create `src/skcomplex/systems/physics/common/quantum/_measures.py` with `OperatorMeasuresResult`,
  `operator_measures`, and their helpers moved from `core/analysis/_operator.py` unchanged in behaviour and wording,
  importing only public core modules, with its own private `_check_choice`, `_check_settings`, source reading, and
  operator problem helper, and a public `DENSE_LIMIT = 1024` in its constant block, verified by ruff, mypy on `src`,
  and lint-imports
- [x] 3.2 Export `operator_measures` and `OperatorMeasuresResult` from `skcomplex.systems.physics.common.quantum` and
  update the package docstring to say it also measures operator systems, verified by the doctests of the package
- [x] 3.3 Delete `src/skcomplex/core/analysis/_operator.py`, remove both names from the imports and `__all__` of
  `skcomplex.core.analysis`, and remove `operator_measures` from the imports and the list of `_discovery.py`, verified
  by ruff, mypy on `src`, and the tests of `tests/core/analysis/test_discovery.py`

## 4. Tests

- [x] 4.1 Move `tests/core/analysis/test_operator.py` to `tests/systems/physics/common/quantum/test_measures.py` with
  its import from the quantum package, verified by the moved tests passing
- [x] 4.2 Change the import of `operator_measures` in `tests/systems/physics/waves/test_quantum.py` and
  `tests/systems/physics/statistical_mechanics/test_quantum_spins.py`, verified by both test modules with `--slow`
- [x] 4.3 Add a test to `tests/core/analysis/test_discovery.py` that `applicable_analyses` of a trajectory of a quantum
  spin chain lists only public names of `skcomplex.core.analysis`, verified by the test passing

## 5. Documentation

- [x] 5.1 Change the import in `docs/overview/user_guide/build_a_custom_system/quantum_systems.md`,
  `docs/examples/physics/waves/plot_schrodinger.py`, and
  `docs/examples/physics/statistical_mechanics/plot_quantum_heisenberg.py`, and say in the guide page where the
  measures live, verified by the markdown tests of the page and ruff and mypy on `docs`
- [x] 5.2 Extend the placement rule in `openspec/constitution.md`, `CONTRIBUTING.md`, and
  `docs/overview/user_guide/extend_the_library.md` with one sentence that analyses follow it by their users, core
  analysis for several fields and `systems/<field>/common/<family>` for one, in the same words in all three, verified
  by the conventions and the markdown tests of the page
- [x] 5.3 Check that no page, docstring, or spec outside `openspec/changes` still imports `operator_measures` from
  `skcomplex.core.analysis`, verified by `rg "core.analysis import[^\n]*operator_measures" src tests docs README.md`
  finding nothing

## 6. Verification and commit

- [x] 6.1 Run ruff on `src`, `tests`, and `docs`, mypy on `src`, `tests`, and `docs`, black on the changed files,
  lint-imports, and the conventions, each clean
- [x] 6.2 Run `tests/core/analysis`, `tests/core/engine/compile`, `tests/core/engine/problems`,
  `tests/systems/physics` with `--slow`, and the doctests of every changed module, each passing
- [x] 6.3 Commit as `refactor!: Move operator_measures to the quantum package of physics` whose footer
  `BREAKING CHANGE:` gives the new import line, and push to `development`

## Workflow follow-up

- Archive the change with `openspec archive move-operator-measures-to-physics -y`, check that the main specs
  validate, commit, and push.
