# Tasks

## 1. Base and specs

- [x] 1.1 Create the branch `restore-abstract-hooks` from `development`, and verify `git status` is clean on it.
- [x] 1.2 In `skcomplex/base.py`, make `check()`, `prepare()`, and `run()` return `Self` and the hooks return `None`, keeping the hooks concrete for now; update the class docstring and doctest; verify `pytest tests/test_base.py --doctest-modules src/skcomplex/base.py` passes.
- [x] 1.3 In `BaseSpec`, implement `_check`, `_prepare`, and `_run` with no effect, keeping the `check()` override that a frozen spec needs since it cannot be marked as checked; verify `pytest tests/test_specs.py --doctest-modules src/skcomplex/specs` passes.

## 2. Source loader

- [x] 2.1 Add `SourceLoader` with `CACHE_DIR` in `skcomplex.engine.problems`, moving the logic of `BaseConfig._load_source`, and make `NumbaBackend`, `ExpressionPrinter`, `LatticeProblemPass`, and `LyapunovAnalysis` use it; remove `_load_source` and `CACHE_DIR` from `skcomplex.base`; verify tests for the loader (a module with the function, the file under the cache directory, injected functions) and `pytest tests/engine tests/analysis/test_lyapunov.py` pass.

## 3. Core

- [x] 3.1 Make `BasePart` and every element, interaction, and system set `laws_` in `_run` and return `None`, and update every caller of their `run()` in `core`, `engine`, `analysis`, `experiments`, and `models`; verify `mypy` on the changed files and `pytest tests/core` pass.
- [x] 3.2 Make every law, including `RenewalLaw`, `ProductionLaw`, and `SubstitutionLaw`, set `value_` in `_run`, and update every caller, such as the relaxation kernel and the renewal solver; verify `mypy` on the changed files and `pytest tests/core/laws` pass.
- [x] 3.3 Make every `_check` and `_prepare` in `core` return `None`; verify `pytest tests/core --doctest-modules src/skcomplex/core` passes.

## 4. Engine

- [x] 4.1 Declare `values_` on `BaseKernel` and narrow it on each kernel, including the fractals' word and patch kernels; make every kernel set it, and update every caller; verify `pytest tests/engine/problems` passes.
- [x] 4.2 Declare `pairs_` on `BaseSearch` and `code_` on `BasePrinter`, narrow them on each search and printer, and update every caller; verify `pytest tests/engine/problems tests/engine/backends` passes.
- [x] 4.3 Declare `kernels_` on `QuantumLowering` and `functions_` on `BaseBackend`, and update every caller; verify `pytest tests/engine/backends tests/engine/compile/passes/test_quantum_compiler.py` passes.
- [x] 4.4 Rename `BaseLowering`'s type parameter to `LoweredT`, declare `lowered_: LoweredT`, and declare `observed_` on `Observer` and the result of `_MemberReductions`; make every lowering and pass set its result, and update every caller; verify `pytest tests/engine/compile` passes.
- [x] 4.5 Declare `problem_` on `BaseCompiler` and `state_` on `BaseSolver`; make every compiler and solver set them, update `Simulation` and the co-simulation to read them without `isinstance` narrowing, and make the named-input `run` overrides of `Simulation`, `BaseObserver`, `BaseClock`, and the Rerun visual forward to the base verb; verify `pytest tests/engine` passes.
- [x] 4.6 Make every remaining `_check` and `_prepare` in `engine` return `None`, and update the engine docstrings and doctests; verify `mypy src/skcomplex/engine` and `pytest tests/engine --doctest-modules src/skcomplex/engine` pass.

## 5. Analysis and experiments

- [x] 5.1 Make every analysis hook return `None`, keep `BaseAnalysis.run` as a forwarding override, and give `_Expansion` a result attribute; verify `pytest tests/analysis --doctest-modules src/skcomplex/analysis` passes.
- [x] 5.2 Declare `rows_` on `BaseDesign`, make every design and experiment hook return `None`, and update every caller; verify `pytest tests/experiments --doctest-modules src/skcomplex/experiments` passes.

## 6. Models

- [x] 6.1 Make every model hook return `None`; make the market protocols keep `trades_` with `_run` returning `None`; update every model that reads a `run()` result; verify `mypy src` and `pytest tests/models --doctest-modules src/skcomplex/models` pass.
- [x] 6.2 Update the user guide, the examples, and the constitution's Project Profile, which says the hooks are abstract and runs keep their results as attributes; verify `pytest docs/overview` passes and `grep` finds no `.run(` call whose result is read as a value in `docs/examples`.

## 7. Abstract hooks

- [x] 7.0 Apply the decisions taken during the migration (design.md): accept `SourceLoader` in the surface test; declare `laws_` on the element, interaction, and system bases only and give `BaseLaw` a `_run` setting `value_`; add `NotPreparedError` and the prepare guard to `BaseConfig`, exempting `BaseSpec`, and reduce `BaseAnalysis`'s overrides to forwarding; verify `pytest tests/test_base.py tests/core tests/engine/problems/test_surface.py tests/analysis/test_base.py` passes.
- [ ] 7.1 Declare `_check`, `_prepare`, and `_run` as abstract methods on `BaseConfig` with complete docstrings, keep `_release` as an optional hook, and implement the hooks a family shares in its base, and the hooks of classes outside a family in the class; verify a test that creates a solver without `_run` raises `TypeError` naming `_run`, and that a spec with no hook checks, prepares, and runs.
- [ ] 7.2 Drop `ABC` from the bases of children that declare no abstract method of their own; verify `mypy src` passes.

## 8. Integration

- [ ] 8.1 Run the checks session (ruff, black, lint-imports, mypy, bandit, interrogate, pydoclint, deptry) and the full suite with doctests, and verify only the known timing test fails.
- [ ] 8.2 Profile a long agent run and a long lattice run on `development` and on the branch, and verify the branch is not slower beyond noise.
- [ ] 8.3 Merge the branch into `development` and delete it, and verify `git branch` no longer lists it.

## Workflow follow-up

- Archive the change with `openspec archive restore-abstract-hooks-and-fitted-results -y` once merged.
