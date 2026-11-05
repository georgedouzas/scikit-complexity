# Proposal

## Why

The documentation does not show the two ways to use the library. When this change was proposed, the user guide had 59
imports from field modules of `skcomplex.models`, and the README built its first model from `PopulationElement` and
`PredationInteraction`, so a reader learned that a custom model is assembled from another model's parts. The intended split is simpler: a custom model is built
from `skcomplex.core`, and a prebuilt model is selected from the catalog.

Nor can a user tell which components fit together. The compatibility page covers laws, solvers, and backends, and nothing
covers elements and interactions. An interaction declares its roles but not what it needs from what fills them, so binding
one to an element without the state it reads raises a bare `AttributeError: ... has no attribute 'n'` from inside
`_get_laws`, while every other mistake gets a report with its path, reason, and fix.

## What Changes

- Organise the README and the user guide around two paths: select a prebuilt model, or build a custom model from the core.
- Examples of custom models import only from `skcomplex.core`, `skcomplex.specs`, `skcomplex.engine`, and
  `skcomplex.analysis`. Prebuilt models are reached through `catalog`, `describe`, `create`, and `run`. A test enforces this.
- Interactions declare what each role needs: the element kinds it accepts, the states it reads, and the time and space its
  laws need. The core exposes this through public class methods, and elements expose the states they carry.
- Validation reports a role filled by an unsuitable element with its path, reason, and fix, in place of the `AttributeError`.
- The generated compatibility page gains elements and interactions, so one page answers what fits what.
- The model catalog in the documentation gains a browsable index by field and tag, and each model page shows how to select
  it and which components it is built from.
- A guide section walks through choosing compatible components, from the time and space to the solver.
- Move the general machinery the guide teaches out of the models and into the core and engine, so a custom model needs no
  field module: rewriting systems, and automata, which become laws and specs rather than a new concept.
- The guide names nothing from a field module, in code or in prose, and leaves `register_model` to the contributing guide.

## Capabilities

### New Capabilities

- `user-documentation`: the two paths through the documentation, and how a user finds compatible components.

### Modified Capabilities

- `system-validation`: interactions declare what their roles need, validation reports a mismatch, and the generated table
  covers elements and interactions.
- `model-catalog`: the documentation offers a browsable catalog, with how to select each model and what it is built from.
- The specs that name these classes follow their move, with the same results.

## Impact

- `skcomplex.specs`: `RoleSpec` gains its requirements. `skcomplex.core`: the 41 interactions and the element kinds declare
  them. `skcomplex.engine.compile`: a new issue, `role-element`.
- `README.md`, `docs/overview/user_guide.md`, `docs/index.md`, `docs/generate_compatibility.py`, `docs/generate_models.py`,
  and `properdocs.yml`. 26 of 577 gallery examples import from field modules, and are checked rather than rewritten.
- Models need no change to their behaviour. Their own interactions may declare their requirements, and those that do not
  still get the report, through the fallback in the design.
- No new package and no new dependency.
