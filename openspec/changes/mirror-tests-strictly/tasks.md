# Tasks

## 1. Examples in the test session

- [x] 1.1 Configure pytest to collect the docstring examples of `src/skcomplex` with `ELLIPSIS`, and point the `tests`
  session at the configured paths. Verify that a planted failing docstring example fails the session, then remove it.
- [x] 1.2 Add `pytest-markdown-docs` to the `tests` dependency group. Run the Python blocks of `README.md` and
  `docs/overview/user_guide.md` with it, tagging blocks that build on earlier ones as continuations and keeping a
  viewer from spawning. Verify that every block runs, and that a planted failing block fails the session.

## 2. Import contracts

- [x] 2.1 Add `import-linter` to the `checks` dependency group, and run it from the `checks` session. Write the
  contracts of the design: core package layers, core without models, shared core without formalisms, formalism
  dependencies, models without private core modules, and self-contained model packages. Verify that `lint-imports`
  passes, and that a planted upward import and a planted formalism import each fail it.

## 3. Inventory check in the docs build

- [ ] 3.1 Move the inventory rules from the deleted test module into `docs/generate_models.py`, and make the
  generator fail with every issue before it writes a page. Verify that the generator passes on the current inventory,
  and that a planted unregistered entry makes it fail with the entry's name.

## 4. Profile and gate

- [ ] 4.1 Update the constitution's Project Profile. Name the gate tools, now including `import-linter` and the
  doctest and Markdown collection, replace the sentence about a layout test with the import contracts, and bump the
  patch version.
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify that all of them pass.
