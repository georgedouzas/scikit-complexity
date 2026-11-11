# Tasks

Every test run below sets `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and uses
`pdm run pytest -n 4`. The import contracts run with `PYTHONPATH=src .nox/checks-3-12-file-src/bin/lint-imports`, and
the conventions with `pdm run python openspec/scripts/check_conventions.py`. Every rename is made by hand per file,
with the rule and the inventory of the design, and never by a replacement across files.

## 1. Public names

- [x] 1.1 Rename `BasePart` to `BaseComponent` in `src/skcomplex/core/components/base.py`, with its docstring, its
  doctest class `DoublingComponent`, and every import and annotation in `src`, `tests`, `docs`, and the
  constitution lines that name the class, verified by `rg -w BasePart src tests docs README.md CONTRIBUTING.md`
  finding nothing, ruff, mypy on `src`, `tests`, and `docs`, lint-imports, and the tests of `tests/core/components`,
  `tests/core/test_base.py`, `tests/core/engine/solvers/test_root.py`, and `tests/core/components/laws/utils`
- [x] 1.2 Rename `GroupElement.get_declaring_part` to `get_declaring_element`, with its return name in the docstring
  and its callers in `core/engine`, `core/analysis`, and the doctests of the Barkley, Gray-Scott, and Schnakenberg
  systems, verified by `rg -w get_declaring_part src` finding nothing, mypy on `src`, and the doctests of those modules
- [x] 1.3 Commit the two renames as one breaking change, `refactor!:`, whose
  footer `BREAKING CHANGE:` says that `BasePart` is now `BaseComponent` and `get_declaring_part` is now
  `get_declaring_element` with no alias, verified by the changelog preview of `git-changelog` listing it under breaking
  changes

## 2. Private names and messages of the core

- [x] 2.1 In `core/components`, rename `_get_part_specs` to `_get_component_specs`, `CompositeSystem._get_parts` to
  `_get_components` with its locals, and reword the docstrings and the messages of `systems/_composite.py`,
  `systems/_base.py`, and `interactions/_base.py`, leaving `PARTS`, `_count_parts`, `symmetric_part`,
  `antisymmetric_part`, `delay_parts`, `Part` of `_coupling.py`, and the string pieces of `_reductions.py`, verified by
  ruff, mypy on `src`, the conventions, and the tests of `tests/core/components`
- [x] 2.2 In `core/engine`, rename `FlattenPass._prepare_part` to `_prepare_component`, the argument `part` of
  `declared_variables`, `declarations_of`, `default_of`, and the helpers of `_utils.py` to `component`, and the locals
  of `_flows.py` and `_census.py`, and say "the components of a vector" in full in the docstrings of the vector helpers
  of `_utils.py`, verified by ruff, mypy on `src`, and the tests of `tests/core/engine`
- [x] 2.3 Rename `_SystemPartsSpec` to `_SystemVariablesSpec`, its module to `_system_variables_spec.py`,
  `_read_parts` to `_read_variables`, and the argument and attribute `parts` to `variables` in
  `core/engine/translators`, verified by ruff, mypy on `src`, lint-imports, and the tests of
  `tests/core/engine/translators`
- [x] 2.4 Reword the docstrings, comments, and messages of `core/engine`, `core/analysis`, `core/experiments`, and
  `core/specs` where part means a component, including `Tags` and its fields, the message of `_stability.py`, and the
  messages of `_utils.py` and `_calibration.py`, and keep every kept meaning of the design, verified by ruff, mypy on
  `src`, the doctests of `src/skcomplex/core`, and the tests of `tests/core/analysis`, `tests/core/experiments`, and
  `tests/core/specs`

## 3. Systems of the fields

- [x] 3.1 Replace `Return the parts of the system` by `Return the components of the system` and `unique among the
  parts of its system` by `unique among the components of its system`, file by file in every field, and give the
  technology and watchmakers systems the summary `Return the elements and interactions of the system`, verified by the
  conventions and the doctests of `src/skcomplex/systems`
- [x] 3.2 Reword the other docstrings of `skcomplex.systems` where part means a component, such as the observables of
  the built components and the hooks of general systems, and keep the domain words of the watchmakers, the technology
  designs, and the fractions of cells, verified by ruff and the doctests of `src/skcomplex/systems`
- [x] 3.3 Run the tests of `tests/systems` in chunks of one field at a time, and verify that all pass

## 4. Tests

- [x] 4.1 Rename `DoublingPart`, `ExtendedPart`, `InvalidSpecPart`, and `SpecifiedPart` in
  `tests/core/components/test_base.py` to the same names with `Component`, and the helper `_parts` of
  `tests/core/engine/translators/test_system_dynamics.py` to `_composite_of`, verified by those modules passing
- [x] 4.2 Rename the test functions and docstrings that say part for a component, such as
  `test_system_checks_every_part` and `test_parts_derive_from_the_part_base`, and update the pinned messages, adding no
  test file and no test that checks an old name is gone, verified by the conventions, which confirm the tests still
  mirror the sources, and the tests of `tests/core` in chunks of one package at a time

## 5. Documentation

- [ ] 5.1 Rename `docs/generate_parts.py` to `docs/generate_components.py`, write the pages to `components/`, rename the
  constants `PARTS` of the generator and of `docs/gallery_hooks.py` to `FAMILIES`, point the links of
  `docs/generate_systems.py` at `components/elements.md`, reword the generated prose, and update `properdocs.yml` with
  the section "Components", verified by the strict docs build
- [ ] 5.2 Add `mkdocs-redirects>=1.2.3` to the `docs` group of `pyproject.toml`, lock it, list `redirects` after
  `gen-files` in `properdocs.yml` with a `redirect_maps` entry for each of the nine family pages and the two moved
  guide pages, and set `validation.links.anchors` to `warn`, verified by the strict docs build and by opening
  `site/parts/elements/index.html#element` in a browser and landing on the `element` section of the new page
- [ ] 5.3 Move `core_components/choosing_parts.md` to `choosing_components.md` and
  `build_a_custom_system/connected_components.md` to `ports_and_connections.md` with the title "Ports and connections",
  and update the nav and every link to them, verified by the strict docs build
- [ ] 5.4 Reword the pages of `docs/overview/user_guide`, the glossary with an entry for component, the
  troubleshooting page, and the gallery examples where part means a component, rename the headings such as "How the
  components fit" and "Where a new component goes", and update every link to their anchors, verified by the markdown
  blocks of `docs/overview/user_guide`, the conventions, and the strict docs build
- [ ] 5.5 Reword the README and CONTRIBUTING, with the link of the README to `components/elements/` and the link of
  CONTRIBUTING to `#where-a-new-component-goes`, verified by the markdown blocks of the README and the conventions

## 6. Specs and the constitution

- [ ] 6.1 Reword the placement rule and the Project Profile of `openspec/constitution.md`, the section "Where a
  component goes" of CONTRIBUTING, `core_components/general.md`, and `extend_the_library.md` so that the four texts of
  the rule are the same, verified by reading them side by side and by the conventions
- [ ] 6.2 Reword the purpose of `openspec/specs/general-systems/spec.md`, which a delta cannot change, verified by
  `openspec validate --specs --strict`
- [ ] 6.3 Check that every main spec requirement that says part for a component is covered by a delta spec of this
  change, adding a delta for any the inventory missed, verified by `openspec validate rename-parts-to-components --strict`

## 7. Drafted changes

- [ ] 7.1 Reword the proposals, designs, tasks, and delta specs of every drafted change under `openspec/changes`,
  archive excluded, where part means a component, such as "a policy as a component of the system" in
  `add-learning-foundation`, verified by `openspec validate <change> --strict` for each changed draft

## 8. Convention check

- [ ] 8.1 Add `check_part_names_are_named` to `SELF_CHECKS`, with the planted module and page of the design, verified
  by the conventions reporting the self-check as passing and by a deliberate break of the reader making it fail
- [ ] 8.2 Add `check_components_are_not_called_parts` to `CHECKS` with `KEPT_PART_NAMES` and `KEPT_PART_PHRASES` built
  from the kept meanings of the design, rename `check_library_generates_no_classes_of_parts` to
  `check_library_generates_no_classes_of_components`, and list the new check in `openspec/scripts/README.md`, verified
  by the conventions reporting no finding on the tree and by ruff and mypy on the script
- [ ] 8.3 Note in the design of this change any kept name or phrase added to the allow lists while running the check,
  with the meaning of the rule it belongs to, verified by review

## 9. Integration

- [ ] 9.1 Run ruff, mypy on `src`, `tests`, and `docs`, lint-imports, and the conventions, and verify all pass with no
  suppression added
- [ ] 9.2 Run the tests in chunks with `-n 4` and the thread caps, `tests/core` package by package, `tests/systems`
  field by field, then the doctests of `src/skcomplex` and the markdown blocks of the README and
  `docs/overview/user_guide`, and verify all pass
- [ ] 9.3 Run the strict docs build and verify it passes, with the nav section "Components", no page left under `parts/`
  but the redirects, and every former URL redirecting
- [ ] 9.4 Run `rg -w -i 'parts?' src tests docs README.md CONTRIBUTING.md openspec/constitution.md` and read every
  remaining match against the rule of the design, verified by each match being a kept meaning. The main specs follow
  at archive, when the delta specs of this change are synced
- [ ] 9.5 Run `openspec validate rename-parts-to-components --strict` and verify it passes
