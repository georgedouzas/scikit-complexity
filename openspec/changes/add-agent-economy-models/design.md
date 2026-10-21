# Design

## Context

The six models were surveyed by `add-agent-economies` and given priority `example`: they use agents in a space,
ordered phases, and markets, which implemented models already exercise.

## Goals / Non-Goals

**Goals:**
- Implement the six models on existing primitives, with tests of their known behaviour and curated examples.

**Non-Goals:**
- New core primitives. A model that turns out to need one is blocked on a change that specifies it.

## Decisions

Each model follows its source section and the patterns of the implemented agent economies, such as `sugarscape` and
`double_auction`. The order follows the inventory, since all six have the same priority.

## Risks / Trade-offs

- [A repeated pattern hides a missing primitive] → The implementation stops and specifies it rather than working
  around it.
