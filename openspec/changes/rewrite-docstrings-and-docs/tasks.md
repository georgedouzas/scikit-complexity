## 1. Preconditions and guard

- [ ] 1.1 Check that `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`, `slim-core-laws`, and `hoist-constants-and-helper-methods` are archived, and verify with `openspec list` that none of them is still open
- [ ] 1.2 Fix the few current summary lines over 80 characters and the docstrings that hold a semicolon in `skcomplex.core`, and verify with an AST walk that none is left
- [ ] 1.3 Add `test_docstring_summaries_are_short` and `test_docstrings_hold_no_semicolon` to `tests/core/components/test_base.py`, walking every module of `skcomplex.core`, and verify they pass
- [ ] 1.4 Add `test_class_summaries_name_their_kind` and `test_docstrings_say_main_public_methods` with the module constant `REWRITTEN`, empty at first, and verify they pass and fail on a planted bad summary
- [ ] 1.5 Add `test_documents_hold_no_semicolon` to `tests/test_sources.py`, fix the findings it reports in the README and the user guide, and verify it passes

## 2. Pilot: elements and the coupling interaction

- [ ] 2.1 Rewrite the docstrings of every module, class, and method of `skcomplex.core.components.elements` and its `__init__` by the rules of design.md, add the cross-domain examples of `NonnegativeElement`, add the package to `REWRITTEN`, and verify the guard and `pytest --doctest-modules src/skcomplex/core/components/elements` pass
- [ ] 2.2 Rewrite the docstring of `CouplingInteraction` and its module, moving mode details to the `Args` entries and fixing the broken list of modes, and verify its doctests and pydoclint pass
- [ ] 2.3 Show the user the diff of 2.1 and 2.2 and record the approved style, or the corrections, in design.md before group 3 starts

## 3. Components

- [ ] 3.1 Rewrite the other modules of `components/interactions`, add the package to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 3.2 Rewrite `components/laws` and `laws/utils`, add them to `REWRITTEN`, and verify the guard and their doctests pass
- [ ] 3.3 Rewrite `components/spaces`, with the neighbourhoods, add it to `REWRITTEN`, and verify the guard and its doctests pass
- [ ] 3.4 Rewrite `components/systems`, `components/specs`, `components/base.py`, and `components/types.py`, add them to `REWRITTEN`, and verify the guard and their doctests pass

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

## 7. Documents

- [ ] 7.1 Pilot: rewrite `docs/overview/user_guide/build_a_custom_system/concepts.md` and `kinetic_systems_of_active_particles.md` by rule 9 of design.md, verify with `pytest-markdown-docs` that every block runs, and show the user the diff before 7.2 starts
- [ ] 7.2 Rewrite the README, and verify its blocks run in the test session and `test_readme_custom_system_runs_from_the_core_alone` passes
- [ ] 7.3 Rewrite the other pages of `docs/overview/user_guide` and `build_a_custom_system`, and verify with `pytest-markdown-docs` that every block runs
- [ ] 7.4 Rewrite the prose that `docs/generate_parts.py` and `docs/generate_systems.py` write, and verify the generated pages build in the documentation session
- [ ] 7.5 Rewrite the docstrings and text cells of the gallery examples under `docs/examples`, keeping each caption of what the screenshot shows, and verify the gallery coverage check passes
- [ ] 7.6 Rewrite `CONTRIBUTING.md`, and verify `test_documents_hold_no_semicolon` and the source tests pass

## 8. Integration

- [ ] 8.1 Remove `REWRITTEN` so the guard checks the whole of `skcomplex.core`, and verify the guard passes
- [ ] 8.2 Run `pdm checks`, the full test session with doctests, and the documentation build, and verify all pass with no suppression added
