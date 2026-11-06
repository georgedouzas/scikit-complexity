# Design

## Context

The package move touches every module, test, example, and document, and the rename from model to system touches most
of the prose. Both have to be mechanical to be safe. See proposal.md for the decisions.

## Goals / Non-Goals

**Goals:**

- One mechanical pass per rename, driven by a script kept out of the repository, then a manual review of what the
  script cannot decide.
- `create_simulation` returns an unprepared `Simulation`, so the catalog never runs anything and the two verbs stay the
  only way to run.

**Non-Goals:**

- Aliases for the old import paths or names.
- Renaming the OpenSpec capability directories, such as `model-catalog`, or editing archived changes and
  `CHANGELOG.md`.

## Decisions

- **The old `core` becomes `skcomplex.core.components`, and the new `skcomplex.core` holds it beside `engine`,
  `analysis`, `experiments`, and the shared `base`, `errors`, `types`, and `specs`.** Merging the old core's `base`,
  `types`, and `specs` with the shared ones was rejected: the part-naming specs and types import `System`, the
  elements, and the spaces, which import the shared ones, so a merge is an import cycle.
- **Relative imports are recomputed, not edited by pattern.** The script resolves each relative import to its old
  absolute module, maps it to the new module, and writes the relative form from the file's new package. Absolute names
  and paths in every text file are mapped by the same table. Files move with `git mv` to keep their history.
- **Model becomes system by a word-level pass with an exception list.** The pass maps `model` to `system`, `models` to
  `systems`, `Model` to `System`, and `MODELS` to `SYSTEMS`, inside identifiers too. It skips identifiers of other
  libraries, such as `mesa.Model` and the COBRA model object, cited titles, and words such as modelling and modeler.
  Where an identifier would collide with an existing `system` in the same scope, such as a test variable, it gets a
  descriptive name by hand.
- **`Simulation` gains `until`.** `create_simulation` stores the registered end time there, so a bare `run()` works.
  The alternative of a catalog function that runs was rejected, since it is a second way to run.
- **The main specs take the rename at archive.** The deltas carry the behaviour changes. After the archive syncs them,
  the same mechanical pass maps paths and the word model in the main specs, as their wording is not behaviour.

## Risks / Trade-offs

- [A string that builds a module name at run time escapes the mapping] → import every module, collect every test, and
  run the catalog conformance tests after the move.
- [The word pass changes a third-party identifier] → mypy, the tests of the external backends and translators, and a
  review of every changed line that touches `mesa`, `cobra`, `pysd`, and SBML.
- [The guide rewrite drops a topic] → compare the section list before and after, and keep every executed block passing.
