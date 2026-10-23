# Proposal

## Why

Four inventory models waited on primitives of changes that are now archived. `szabo_cells` and
`active_passive_flocking` waited on the neighbourhood reductions and network generators of `add-adaptive-networks`,
and `koopman_model_predictive_control` and `next_generation_reservoir` waited on the learned surrogates of
`add-data-driven-dynamics`, which `add-optimal-control` complements with receding-horizon control. Those changes
added the primitives but not these models, so the models have no change that builds them.

## What Changes

- Add `szabo_cells`, `active_passive_flocking`, `koopman_model_predictive_control`, and `next_generation_reservoir`
  to the library, each a registered factory under its field path with a curated gallery example and a review row.
- Correct the system tags of each entry to the system the model builds, and mark the entries implemented.
- Write any field-specific need in the field's model library, never in the core.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: four models built on primitives that are now available.

## Impact

The model library, its tests, the gallery, and the inventory. No core primitive changes.
