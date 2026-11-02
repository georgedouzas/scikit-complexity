# Design

## Context

See proposal.md for the motivation.

The maintainer deleted 20 files from `tests`, holding 2 692 lines. The tests they removed did four jobs:
- **Conventions:** `test_layout.py` and `layout_checks.py` checked package import order, formalism imports, model
  imports, definition order, guarded imports, export leaks, and test mirroring with a homegrown AST checker.
  `test_agnostic_core.py` checked that the core runs without the models.
- **Docstring examples:** `test_docstrings.py` walked every module and ran `doctest.testmod` with `ELLIPSIS`.
- **Documentation examples:** `test_documentation.py` ran the ```` ```python ```` blocks of `README.md` and
  `docs/overview/user_guide.md` in order, as one module, with `rerun.spawn` patched out.
- **Documentation tooling and data:** `tests/docs` tested `docs/gallery_tools`, `docs/generate_compatibility.py`, the
  gallery hooks, and the inventory rules. The inventory rules lived only in that test module.

pytest runs `tests` only (`testpaths = ["tests"]`, `addopts = "--cov=src --cov=tests"`), and `nox -s tests` calls
`pytest -n auto tests`.

## Goals / Non-Goals

**Goals:**
- A test tree that mirrors `src/skcomplex` exactly.
- Each job of the deleted tests done by the tool built for it, configured once in `pyproject.toml`.

**Non-Goals:**
- Tests for `docs/gallery_tools`. They are documentation tooling, and the docs build exercises them.
- A replacement for the definition-order and export checks.

## Decisions

### Doctest collection replaces the docstring test module

pytest runs the doctests with `--doctest-modules` on `src/skcomplex`, with `doctest_optionflags = ELLIPSIS` and
`testpaths = ["tests", "src/skcomplex"]`. The `tests` session runs `pytest -n auto`, so the configured test paths
apply.
- **Alternative: keep a parametrized test module.** It was rejected because it breaks the mirror.

### pytest-markdown-docs runs the document blocks

`pytest-markdown-docs` collects fenced Python blocks of the Markdown files given to it with `--markdown-docs`. It is a
mature plugin, and it reports failures by file and line. Its settings:
- **Inputs:** the `tests` session passes `README.md` and `docs/overview/user_guide.md`.
- **Continuations:** a block that reads names from earlier blocks gets the `continuation` info string, as in
  ```` ```python continuation ````. That is how the old module ran the blocks in order.
- **Viewer:** `rerun.spawn` is patched in a `conftest.py` fixture that the plugin reads (`pytest_markdown_docs_globals`),
  or the guide's blocks write recordings instead of spawning a viewer.
- **Alternative: markdown-exec in the docs build.** It was rejected by the maintainer in favour of this plugin.

### import-linter enforces the import rules

`import-linter` reads contracts from `[tool.importlinter]` in `pyproject.toml`, and the `checks` session runs
`lint-imports` next to ruff and mypy. The contracts:
- **Core packages:** a `layers` contract over the core packages, in their order. Lower packages may not import higher
  ones.
- **Models:** a `forbidden` contract: `skcomplex.core` may not import `skcomplex.models`.
- **Formalisms:** a `forbidden` contract: the shared core packages may not import `skcomplex.core.formalisms`. Then one
  `forbidden` contract per formalism, forbidding the formalisms it does not declare in `depends`.
- **Private core modules:** a `forbidden` contract: `skcomplex.models` may not import `skcomplex.core.**._*` modules.
  This uses wildcards. If the installed version cannot express it, models are checked against the public surface by
  `forbidden` contracts per core package.
- **Model packages:** an `independence` contract over the subfield packages of each field, plus `forbidden` contracts
  between fields. The realizations of each field are allowed.

The `layers` contract also covers the base-class modules. `components` and `laws` import only lower packages, and
`base` is the lowest.
- **Alternative: ruff banned-api.** It was rejected because it cannot express the package order.

### The docs build runs the inventory check

The inventory rules move from the deleted test module into `docs/generate_models.py`:
- the entry shape;
- agreement with the catalog;
- known primitives;
- vocabulary mapping;
- priorities;
- statuses that name existing changes.

`main()` collects every issue before it writes a page, and raises one error that lists them. The docs build runs
`generate_models.py` through the `gen-files` plugin, so a broken inventory fails the build. The code is restored from
the deleted module, `git show 1df3e17:tests/docs/overview/test_inventory.py`.

## Risks / Trade-offs

- [Continuation tags change how the guide reads] → The tags sit in the fence info string, which the rendered page does
  not show.
- [`--doctest-modules` imports every module of the package] → The deleted test did the same, so the cost does not
  change.
- [The definition order is no longer checked] → The constitution binds it, and reviews check it.

## Migration Plan

1. Change the pytest configuration and the dependencies, then the guide's fences, and run the test session.
2. Write the import contracts, run `lint-imports`, and fix any violation it finds.
3. Move the inventory check into the docs generator.
4. Update the Project Profile.
5. Run the gate.
