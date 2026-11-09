## 1. Preconditions

- [x] 1.1 Check that `slim-core-elements-and-spaces`, `slim-core-interactions`, and `slim-core-laws` are archived and that no other change is being applied, and verify `openspec list` shows none of the three and `git status` is clean
- [x] 1.2 Rebuild the `General elements` block of `specs/general-mechanisms/spec.md` in this change from the archived main spec, renaming only `MassActionSystem`, recheck every other MODIFIED block against its main spec, and verify `openspec validate rename-core-systems --strict` passes

## 2. Core systems package

- [x] 2.1 `git mv` `core/components/systems/_system.py` to `_composite.py` and rename the class to `CompositeSystem(BaseSystem['CompositeSystem'])`, update its docstrings, its doctests, and its own `isinstance` check, and verify `pytest --doctest-modules src/skcomplex/core/components/systems` passes
- [x] 2.2 `git mv` `_mass_action.py` to `_reaction_network.py` and rename the class to `ReactionNetworkSystem(BaseSystem[CompositeSystem])`, rewrite its module and class docstrings as a system of species and reactions whose propensity is a rate constant times the product of the reactant amounts, each raised to its stoichiometric coefficient, naming the law of mass action in chemistry, homogeneous mixing in epidemiology, and the encounter term of Lotka-Volterra in ecology only as examples, and verify its doctests pass
- [x] 2.3 Update `systems/__init__.py`, the docstrings and the error message of `_base.py` that name `System`, and the layers contract on `skcomplex.core.components.systems` in `pyproject.toml` to `_reaction_network`, `_composite`, `_base`, and verify `lint-imports` passes with `PYTHONPATH=src`
- [x] 2.4 `git mv` `tests/core/components/systems/test_system.py` to `test_composite.py` and `test_mass_action.py` to `test_reaction_network.py`, rename the classes in them and in `test_base.py`, add a test that importing `System` or `MassActionSystem` from the package raises `ImportError`, and verify `pytest tests/core/components/systems` passes

## 3. Product propensity of reactions

- [x] 3.1 In `core/components/interactions/_reaction.py` rename `_get_mass_action` to `_get_product_propensity`, and rewrite the module, class, argument, and method docstrings to describe the default propensity as the rate constant times the product of the reactant and modifier amounts, each raised to its coefficient, with the field terms only as examples, and verify its doctests pass
- [x] 3.2 Rename the calls of `_get_mass_action` in `systems/biology/epidemiology/_factories.py` and `tests/core/components/interactions/test_reaction.py`, add a test in `test_reaction.py` that the default propensity of `2 A -> B` at the rate constant `k` is `k A^2`, and verify `grep -rn "_get_mass_action" src tests docs` finds nothing and `pytest tests/core/components/interactions/test_reaction.py tests/systems/biology/epidemiology` passes

## 4. Rest of the core

- [x] 4.1 Rename `System` in the imports, `isinstance` checks, type hints, error messages, and doctests of `core/analysis` (`_base.py`, `_automata.py`, `_stability.py`, `_invariants.py`, `_elementary_modes.py`), `core/engine/runtime/_simulate.py`, `core/engine/compile/_base.py`, `core/engine/compile/passes/_flatten.py`, and every other core module that imports it, and verify `pytest --doctest-modules src/skcomplex/core` and the core tests pass
- [x] 4.2 Make `core/engine/translators/_module_writer.py` import `CompositeSystem` and emit `CompositeSystem(` in the generated `_build_system`, update the docstrings of `_system_dynamics.py`, update the translator tests that compare generated source, and verify `pytest tests/core/engine/translators` passes
- [x] 4.3 Rewrite every "mass action" and "mass-action" in `src/skcomplex/core` as the product propensity, or keep it only as a named field example, and verify `grep -rniE "mass[ -]action" src/skcomplex/core` shows only such examples

## 5. Systems library

- [ ] 5.1 With a throwaway script outside the repository, rename the imported name `System` and its bare uses, never a name joined to a word character or a dot, in every module of `src/skcomplex/systems` that imports it from the systems package, in code and doctests, and verify `git diff --stat` touches no module that does not import it
- [ ] 5.2 Rename `MassActionSystem` in `systems/chemistry/kinetics/_reactions.py` and `systems/biology/epidemiology/_factories.py`, and verify `grep -rn "MassActionSystem" src` finds nothing
- [ ] 5.3 Review every remaining hit of the bare word `System` in `src` by hand, rename those that name the core class, keep prose and other names, and verify `pytest --doctest-modules src/skcomplex/systems` and `pytest tests/systems` pass with `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`, and the BLAS threads set to 1

## 6. Tests

- [ ] 6.1 Apply the same rewrite to every test module that imports `System`, update tests that match an error message naming `System`, review the remaining hits of the bare word by hand, and verify `pytest tests` passes

## 7. Documentation and configuration

- [ ] 7.1 Apply the rewrite to the user guide pages, the gallery examples, `docs/generate_parts.py`, and `docs/generate_systems.py`, keeping table headers and prose such as "System page", and verify the generators run and the docs build passes in strict mode
- [ ] 7.2 Rewrite "mass action" and "mass-action" in the user guide and in `docs/generate_parts.py` as reaction networks with a product propensity, keeping the term only as a named field example, and verify `grep -rniE "mass[ -]action" docs/overview docs/generate_parts.py` shows only such examples
- [ ] 7.3 Update the README and CONTRIBUTING, and the extension points of the profile in `openspec/constitution.md`, and verify `grep -rnwE "System|MassActionSystem" README.md CONTRIBUTING.md openspec/constitution.md` shows no reference to the old core classes

## 8. Verification

- [ ] 8.1 Grep src, tests, docs, the README, CONTRIBUTING, and `pyproject.toml` for `MassActionSystem`, `_get_mass_action`, `_system`, `_mass_action`, `test_system`, and `test_mass_action`, and for the bare word `System`, and verify every remaining hit is reviewed prose or another name
- [ ] 8.2 Run `nox -s checks` for `src`, `tests`, and `docs`, including ruff, mypy, the import contracts, the full test session with doctests, and the strict docs build, and fix every finding without a suppression

## Workflow follow-up

- Archive the change with `openspec archive rename-core-systems`, which applies the delta specs to the main specs.
- List both renames and the method rename in the changelog of the next release.
