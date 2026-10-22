# Proposal

## Why

Every textbook in `references/textbooks` is cited by the inventory, but eight are cited by four entries or fewer:
Barsky 2020, Mihailović et al. 2023, and Starkovich with two each, Mobus 2022, Satz 2022, and Konieczny et al. 2023 with
three, and Gooding 2019 and Schwichtenberg with four. Some are conceptual books with few runnable models, and some may
simply be under-surveyed. The papers in `references/papers` have not been surveyed at all. Since models exist first to
probe the core, an unsurveyed source can hide a missing primitive.

**Rebased on `redesign-system-model`.** The redesign replaced the inventory's flat primitive list with the categories of
the system vocabulary, time kind, space kind, component kinds, interaction bindings, and law classes, and its inventory
check fails on any unmapped entry. This change keeps its scope, the survey, and records each new entry in those
categories. A need that does not map is resolved by a general primitive of the core, while a field-specific need is a
realization in the model library and not a core primitive.

## What Changes

- Survey the eight thinly cited textbooks section by section, and record every model or measurable construction they
  describe in the inventory with its sources, field path, its tags in the system vocabulary, and priority, or record that a section has none.
- Survey every paper in `references/papers` the same way, citing it by arXiv identifier.
- Route every entry that does not map to the system vocabulary to the pending change that adds the general primitive
  it needs, or propose a new change. Route an entry that needs only a field-specific realization to the model library
  of its field.
- Record the survey of each source in the change's design, so a later survey does not repeat it.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-inventory`: every source in the references is surveyed section by section, with the survey recorded.

## Impact

- `docs/overview/inventory.toml` gains entries tagged in the system vocabulary, and possibly vocabulary terms that a
  proposed change adds; no code changes.
