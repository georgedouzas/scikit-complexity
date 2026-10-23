# Design

## Context

`survey-thin-references` recorded each surveyed source section by section and added its models to the inventory. The
models that map to existing vocabulary terms and primitives wait on this change; those that need a primitive of
another pending change are blocked on that change.

## Goals / Non-Goals

**Goals:**
- Implement every surveyed model the core expresses, `core` entries first.

**Non-Goals:**
- New core primitives. A model that turns out to need one stops, and its need becomes a change of its own.

## Decisions

### Order

The `core` entries go first: `directed_percolation_langevin`, `closed_contour_cell_exchange`, `benard_marangoni`,
`metaphoria`, `wilhite_trade_networks`, and the other entries the inventory marks `core`. They exercise combinations
of primitives no implemented model exercises, so they test the core where it is least tested.

### Mass-conserving noise (resolved)

`active_model_b` and `cahn_hilliard` with noise need noise that conserves the field, the divergence of a random flux.
Before implementing them, the change checks whether `Noise` on a field with a `Flux` law covers it; if not, the need
becomes a change of its own and the two entries are blocked on it. The core has no such noise, so `active_model_b`
moved to `add-conserved-noise`, and `cahn_hilliard` is implemented without noise, as its inventory entry records.

## Risks / Trade-offs

- [Sources describe some mechanisms only qualitatively] → The model follows the standard equations of the mechanism
  the source names, and cites them.
