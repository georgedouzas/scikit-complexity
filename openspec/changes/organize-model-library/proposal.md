# Proposal

## Why

The model library is the main way users learn the general interface: every ready-made model shows how to write a model
with it. Today the library has three problems:
- **No line between the general framework and domain models.** Nine domain-agnostic packages sit beside `models` at
  the top of `skcomplex`: `core`, `model`, `compile`, `solvers`, `runtime`, `viz`, `experiments`, `analysis`, and
  `backends`.
- **Packages mix organizing principles.** Inside `models`, scientific fields (`ecology`, `physics`) sit beside
  mathematical techniques (`lattices`, `fields`, `nonlinear`) and application domains (`fluids`), so a reader cannot
  find a model by science.
- **Coverage is thin.** The library has about 80 models. A survey of the 28 reference books and 15 arXiv reviews found
  about 230 distinct models.

The survey also showed which models the interface cannot express yet. Those gaps should become specified features, not
workarounds hidden in model code.

## What Changes

- **BREAKING:** move every domain-agnostic package under `skcomplex.core`, leaving `skcomplex.models` as its only
  sibling:
  - today's `core` becomes `skcomplex.core.base`;
  - `model`, `compile`, `solvers`, `runtime`, `viz`, `experiments`, `analysis`, and `backends` become
    `skcomplex.core.<package>`;
  - the core never imports `skcomplex.models`.
- **BREAKING:** reorganize `skcomplex.models` into a hierarchy of scientific fields and subfields:
  - `physics`, `chemistry`, `biology`, `earth`, `economics`, `social`, `engineering`, `mathematics`, and
    `computer_science`;
  - each field has subfield packages, such as `physics.fluids`, `biology.ecology`, and `economics.finance`;
  - existing models move to their scientific home, for example fluids into `physics.fluids` and Gray-Scott into
    `chemistry.pattern_formation`;
  - the old package paths are removed.
- **BREAKING:** catalog entries carry a field path, `biology/ecology`, and the catalog filters by field and subfield.
- Model modules import only the public names of `skcomplex.core`, together with third-party numerical libraries.
  Model packages are orthogonal and self-contained, and none imports another. A component that several packages need
  is domain-agnostic and moves into the core.
- Every model's documentation shows its laws and equations as written with the interface, so it serves as a pattern
  for user models.
- Expand the library to every surveyed model the interface supports. The inventory in `design.md` lists about 150 new
  models, from celestial mechanics and active matter to gene circuits, climate energy balance, econophysics, opinion
  dynamics, traffic, and network science.
- Add a model inventory, published in the documentation. It records every surveyed model with its source, field,
  status (implemented, pending a planned change, or blocked by a missing feature), and the change that would unblock
  it.
- Add a rule for implementation: when a model needs a feature the interface lacks, the feature is specified as its own
  OpenSpec change and the model is recorded as blocked. Models MUST NOT work around the interface.
- Stand on the shoulders of giants:
  - a missing feature that a popular, mature, or performant library implements is specified as a wrapper around that
    library, not as new numerics;
  - the framework's own numerics are audited against established libraries, and replacements become follow-up changes;
  - the constitution gains this rule.

## Capabilities

### New Capabilities

- `package-layout`: the split of `skcomplex` into the domain-agnostic core and the domain model library, and the
  import direction between them.
- `model-inventory`: the inventory of surveyed models with their sources, fields, and status. It is published in the
  documentation and checked against the catalog, and it records missing interface features as specified changes.

### Modified Capabilities

- `model-library`: field and subfield hierarchy, catalog field paths, models built only from the core, documentation as
  patterns, and the expanded model collections of every field.

## Impact

- **Import paths:** every import path in the package changes, both the framework's (`skcomplex.model` becomes
  `skcomplex.core.model`) and the models'. So do the tests, user guide, README, gallery, docstring examples, catalog
  field names, and the layering in the constitution.
- **Pending changes:** the nine pending changes that add models will use the new paths, place their models in the new
  hierarchy, and record them in the inventory.
- **Follow-up changes:** gaps found by the survey become follow-up changes created during implementation. The expected
  ones are:
  - stochastic discrete maps;
  - anomalous stochastic processes;
  - complex-valued states;
  - nonlocal field operators;
  - lattice geometries and extremal updates;
  - phase-space fields;
  - tensor fields;
  - extended agent spaces.
- **Constitution:** the body gains the rule to build on mature libraries, and the Project Profile gains `scikit-fem`,
  the core and models layout, and the layering inside the core.
