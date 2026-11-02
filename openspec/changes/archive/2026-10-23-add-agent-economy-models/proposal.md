# Proposal

## Why

`add-agent-economies` added phases, markets, institutions, and learning to agent models, implemented the models that
exercise them in new ways, and left six models pending because they repeat patterns those models already cover. That
change is archived, so the six entries need a change that implements them.

**Rebased on `redesign-system-model`.** The redesign turned the primitives these models were to use into general core
primitives and library realizations: phases are the schedule `DiscreteTime(schedule=Phases([...]))`, agents are
elements with `Rule` laws in variable groups on a `LatticeSpace`, `NetworkSpace`, or `ContinuousSpace`, trades are
`Match` and `Transfer` laws with `Invariant` checks, institutions are `parameter(adaptive=True)`, and markets, order
books, and institutions are realizations in `skcomplex.models.economics.realizations`. This change still adds the six
models on that vocabulary; it adds no core primitive.

## What Changes

- Add `goods_traders`, `shops`, `barter`, `order_book`, `tragedy_of_the_commons`, and `ridesharing` as registered
  factories, each with a curated gallery example. Each is a `System` with discrete time and a phases schedule, agent
  groups with `Rule` laws, a space where the source has one, and the library's market realizations over `Match`,
  `Transfer`, and `Invariant`. Each catalog entry declares its standard observables.
- Mark their inventory entries implemented, tagged with the categories of the new vocabulary.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: six agent economy models that repeat covered patterns.

## Impact

- `skcomplex.models.economics` and `skcomplex.models.social` gain six factories; no core change. Shared trading
  realizations come from `skcomplex.models.economics.realizations`; a social model that needs one defines its own.
