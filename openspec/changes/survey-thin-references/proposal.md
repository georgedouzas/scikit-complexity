# Proposal

## Why

Every textbook in `references/textbooks` is cited by the inventory, but eight are cited by four entries or fewer:
Barsky 2020, Mihailović et al. 2023, and Starkovich with two each, Mobus 2022, Satz 2022, and Konieczny et al. 2023 with
three, and Gooding 2019 and Schwichtenberg with four. Some are conceptual books with few runnable models, and some may
simply be under-surveyed. The papers in `references/papers` have not been surveyed at all. Since models exist first to
probe the core, an unsurveyed source can hide a missing primitive.

## What Changes

- Survey the eight thinly cited textbooks section by section, and record every model or measurable construction they
  describe in the inventory with its sources, field path, primitives, and priority, or record that a section has none.
- Survey every paper in `references/papers` the same way, citing it by arXiv identifier.
- Route every entry that needs a missing primitive to the pending change that adds it, or propose a new change.
- Record the survey of each source in the change's design, so a later survey does not repeat it.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-inventory`: every source in the references is surveyed section by section, with the survey recorded.

## Impact

- `docs/overview/inventory.toml` gains entries and possibly primitives; no code changes.
