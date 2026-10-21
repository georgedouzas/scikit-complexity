# Proposal

## Why

`add-agent-economies` added phases, markets, institutions, and learning to agent models, implemented the models that
exercise them in new ways, and left six models pending because they repeat patterns those models already cover. That
change is archived, so the six entries need a change that implements them.

## What Changes

- Add `goods_traders`, `shops`, `barter`, `order_book`, `tragedy_of_the_commons`, and `ridesharing` as registered
  factories on the existing phase, market, and agent-space primitives, each with a curated gallery example.
- Mark their inventory entries implemented.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `model-library`: six agent economy models that repeat covered patterns.

## Impact

- `skcomplex.models.economics` and `skcomplex.models.social` gain six factories; no core change.
