## 1. Preconditions and guard

- [x] 1.1 Check that `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`, `slim-core-laws`, `rename-core-systems`, `move-recipes-to-systems`, and `hoist-constants-and-helper-methods` are archived, and verify with `openspec list` that none of them is still open
- [x] 1.2 Fix the current summary lines over 80 characters and the docstrings that hold a semicolon, a dash used as punctuation, or a banned phrase in `skcomplex.core`, and verify with an AST walk that none is left
- [x] 1.3 Add `BANNED_PHRASES`, `test_docstring_summaries_are_short`, and `test_docstrings_read_naturally` to `tests/core/components/test_base.py`, walking every module of `skcomplex.core`, and verify they pass and fail on a planted em dash, spaced hyphen, and banned phrase
- [x] 1.4 Add `test_class_summaries_name_their_kind`, `test_docstrings_say_main_public_methods`, and `test_components_compare_and_name_uses` with the module constant `REWRITTEN`, empty at first, and verify they pass and fail on a planted bad summary
- [x] 1.5 Add `test_documents_read_naturally` with `REWRITTEN_DOCUMENTS`, empty at first, and the test that keeps its banned list equal to `BANNED_PHRASES`, to `tests/test_sources.py`, and verify they pass and fail on a planted dash in a listed page
- [x] 1.6 Add `test_documents_show_only_public_names` to `tests/test_sources.py`, which fails on a call or attribute read of an underscore name in any docstring example of `src` and any code block of the README, the user guide, and the gallery, allowing only `def _hook` inside a class body. Rewrite every current offender to use the public interface (the trajectory, `laws_` after `run`, results) or move its check into the tests, and verify the test passes and fails on a planted `system._get_laws(namespace)`

## 2. Pilot: elements and the coupling interaction

- [x] 2.1 Rewrite the docstrings of every module, class, and method of `skcomplex.core.components.elements` and its `__init__` by the rules of design.md, with the "Unlike" comparison and the "It appears in" uses paragraph of every class, add the cross-domain examples of `DeclaredElement`, add the package to `REWRITTEN`, and verify the guard and `pytest --doctest-modules src/skcomplex/core/components/elements` pass
- [x] 2.2 Rewrite the docstring of `CouplingInteraction` and its module, moving mode details to the `Args` entries, fixing the broken list of modes, and adding its comparison and uses paragraphs, and verify its doctests and pydoclint pass
- [ ] 2.3 Read the rewritten docstrings through for flow and rewrite what sounds generated, then show the user the diff of 2.1 and 2.2 and record the approved style, or the corrections, in design.md before group 3 starts

## 3. Components

- [ ] 3.1 Rewrite the other modules of `components/interactions`, add the package to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 3.2 Rewrite `components/laws` and `laws/utils`, add them to `REWRITTEN`, and verify the guard and their doctests pass
- [ ] 3.3 Rewrite `components/spaces`, with the neighbourhoods, add it to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 3.4 Rewrite `components/systems`, `components/specs`, `components/base.py`, and `components/types.py`, add them to `REWRITTEN`, and verify the guard and their doctests pass
- [ ] 3.5 Rewrite the docstrings of `skcomplex.systems.general` by the same rules, with the comparison paragraph naming the core parts each recipe is written with, such as "Unlike `JumpInteraction`" for `ReactionInteraction`, including the drafts of `NonnegativeElement` and `EncounterInteraction` from the first version of design.md, add the package to the walk of the guard and to `REWRITTEN`, and verify the guard and `pytest --doctest-modules src/skcomplex/systems/general` pass

## 4. Specs and root modules

- [ ] 4.1 Rewrite `skcomplex.core.specs`, add it to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 4.2 Rewrite `skcomplex.core.base`, `errors`, and `types`, add them to `REWRITTEN`, and verify the guard and their doctests pass

## 5. Analysis and experiments

- [ ] 5.1 Rewrite `skcomplex.core.analysis`, functions and result classes, with uses from two or more fields per function, add it to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 5.2 Rewrite `skcomplex.core.experiments` and its specs, add it to `REWRITTEN`, and verify the guard and its doctests pass

## 6. Engine

- [ ] 6.1 Rewrite `engine/runtime`, `engine/viz`, `engine/backends`, `engine/translators`, and the engine root modules, add them to `REWRITTEN`, and verify the guard and their doctests pass
- [ ] 6.2 Rewrite `engine/solvers`, naming uses where a user picks a solver by hand, add it to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 6.3 Rewrite `engine/compile`, add it to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 6.4 Rewrite `engine/problems`, add it to `REWRITTEN`, and verify the guard and its doctests pass

## 7. Documents pilot: how the parts fit, the elements reference, and the build tutorial

- [ ] 7.1 Check that `add-identifiability-analysis` is archived before the documents start, and verify with `openspec list`
- [ ] 7.2 Add the Mermaid fence to `properdocs.yml`, name `griffelib` in the docs group of `pyproject.toml`, and verify `pdm lock --check` and a strict documentation build pass
- [ ] 7.3 Add the comparison table and the fact sheets of elements to `docs/generate_parts.py`, reading the docstrings with griffe and the tags from the code, with links to the reference, the API, and the catalog systems that use each class, and verify the strict build shows a fact sheet for every element class
- [ ] 7.4 Add the links from catalog system pages to the core classes they build from in `docs/generate_systems.py`, and verify the page of `SIRSystem` links its element classes in a strict build
- [ ] 7.5 Write `core_components/index.md` with the two Mermaid diagrams, the worked example built step by step, and the import table, merging `concepts.md` and updating every link to it, and verify the diagrams render in a strict build and every block runs with `pytest-markdown-docs`
- [ ] 7.6 Write `core_components/elements.md` by the section template of design.md for every public element class, and verify every block runs with `pytest-markdown-docs`
- [ ] 7.7 Add `test_core_components_pages_cover_every_class` to `tests/test_sources.py`, checking the elements page only at first and rejecting an import of `skcomplex.systems` on a family page, and verify it passes, fails when a section is removed, and fails on a planted import of `NonnegativeElement`
- [ ] 7.8 Rewrite `build_a_custom_system/index.md` as the build tutorial with the hare and lynx running example and its cited data, and verify every block runs with `pytest-markdown-docs` in under a minute
- [ ] 7.9 Update the navigation in `properdocs.yml` for the pages of this group, add them to `REWRITTEN_DOCUMENTS`, read each page through for flow and rewrite the sentences that sound generated, and verify `test_documents_read_naturally` and the strict build pass
- [ ] 7.10 Show the user the elements reference, its fact sheets, the page on how the parts fit, and the build tutorial, and record the approved style, or the corrections, in design.md before group 8 starts

## 8. The rest of the reference

- [ ] 8.1 Write `core_components/interactions.md` and generate its table and fact sheets, and verify the strict build, `pytest-markdown-docs`, and the coverage test of the page pass
- [ ] 8.2 Write `core_components/laws.md` and generate its table and fact sheets, and verify the strict build, `pytest-markdown-docs`, and the coverage test of the page pass
- [ ] 8.3 Add the generated page `parts/spaces.md` for spaces, neighbourhoods, and times with its navigation entry, write `core_components/spaces.md`, and verify the strict build, `pytest-markdown-docs`, and the coverage test of the page pass
- [ ] 8.4 Write `core_components/systems.md` and generate its table and fact sheets, and verify the strict build, `pytest-markdown-docs`, and the coverage test of the page pass
- [ ] 8.5 Write `core_components/general.md` on `skcomplex.systems.general`, starting from the page on shared realizations that `move-recipes-to-systems` writes, with one section per public class of its concept subpackages that names the core parts it is written with, and the placement rule word for word with one worked example per step, extend the coverage test to the page, and verify the strict build, `pytest-markdown-docs`, and the coverage test pass
- [ ] 8.6 Write `core_components/choosing_parts.md`, the decision guide with Mermaid flow charts whose leaves link reference sections, and verify the strict build passes
- [ ] 8.7 Add the pages of this group to `REWRITTEN_DOCUMENTS`, read each page through for flow and rewrite what sounds generated, and verify `test_documents_read_naturally` passes

## 9. Workflow tutorials

- [ ] 9.1 Write `check.md`, merging `choosing_compatible_components.md` and updating every link to it, with the common validation errors triggered, printed, and fixed, and verify every block runs
- [ ] 9.2 Rewrite `simulate.md` as a tutorial on the running example covering run settings, solvers, backends, and `random_state`, and verify every block runs
- [ ] 9.3 Rewrite `analyse.md` as a tutorial on the running example, starting from `applicable_analyses`, and verify every block runs
- [ ] 9.4 Write `evaluate.md` with the comparison to the pelt records, goodness of fit, posterior predictive checks, `posterior_coverage`, and `identifiability`, and verify every block runs in under a minute
- [ ] 9.5 Write `calibrate.md` with every method of `calibrate`, its losses, diagnostics, and when to choose it, using small budgets and fixed seeds, and verify every block runs in under a minute and prints rounded or compared values only
- [ ] 9.6 Rewrite `run_experiments.md` as a tutorial covering `sweep`, `ensemble`, `sensitivity`, `compare_scenarios`, `basin_stability`, and `orbit_diagram`, and verify every block runs in under a minute
- [ ] 9.7 Rewrite `observe.md` as the visualise tutorial with the `RerunObserver`, the visuals, and matplotlib plots, and verify every block runs
- [ ] 9.8 Write `share.md` on turning the example into a registered system with its inventory entry and gallery example, linking `CONTRIBUTING.md`, and verify every block runs and the strict build passes
- [ ] 9.9 Update the navigation to the order of the modified requirement, add the pages to `REWRITTEN_DOCUMENTS`, read each page through for flow and rewrite what sounds generated, and verify `test_documents_read_naturally` and the strict build pass

## 10. Case studies and extras

- [ ] 10.1 Write `case_studies/epidemic_calibration.md` on the 1978 boarding school influenza outbreak, citing the report and holding the counts inline, and verify every block runs in under a minute
- [ ] 10.2 Write `case_studies/opinion_sweep.md` on a bounded confidence model swept over its bound, and verify every block runs in under a minute
- [ ] 10.3 Write `case_studies/ising_transition.md`, locating the susceptibility peak near the exact critical temperature and estimating exponents, and verify every block runs in under a minute and the printed temperature agrees with Onsager's value within the stated tolerance
- [ ] 10.4 Write `glossary.md` with one entry per mathematical and library term the guide uses, linked from the pages that use it, and verify the strict build passes
- [ ] 10.5 Write `troubleshooting.md` with one section per public error class of `skcomplex.core.errors`, each raising the error and showing the fix, add `test_troubleshooting_covers_every_error`, and verify both pass
- [ ] 10.6 Add the pages of this group to `REWRITTEN_DOCUMENTS`, read each page through for flow and rewrite what sounds generated, and verify `test_documents_read_naturally` passes

## 11. Other documents

- [ ] 11.1 Rewrite the README, and verify its blocks run in the test session and `test_readme_custom_system_runs_from_the_core_alone` passes
- [ ] 11.2 Rewrite the other pages of `docs/overview/user_guide` and the recipes of `build_a_custom_system`, linking each core class to its reference section and each term to the glossary, and verify with `pytest-markdown-docs` that every block runs
- [ ] 11.3 Rewrite the other prose that `docs/generate_parts.py` and `docs/generate_systems.py` write, and verify the strict build passes
- [ ] 11.4 Rewrite the docstrings and text cells of the gallery examples under `docs/examples`, keeping each caption of what the screenshot shows, and verify the gallery coverage check passes
- [ ] 11.5 Rewrite `CONTRIBUTING.md`, add the documents of this group to `REWRITTEN_DOCUMENTS`, read each through for flow, and verify `test_documents_read_naturally` passes

## 12. Integration

- [ ] 12.1 Remove `REWRITTEN` and `REWRITTEN_DOCUMENTS` so the guards check the whole core and every document, and verify they pass
- [ ] 12.2 Run `pdm checks`, the full test session with doctests, and the documentation build, and verify all pass with no suppression added
