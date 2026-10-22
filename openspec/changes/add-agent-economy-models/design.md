# Design

## Context

The six models were surveyed by `add-agent-economies` and given priority `example`: they use agents in a space,
ordered phases, and markets, which implemented models already exercise. After `redesign-system-model`, those patterns
are written with general core primitives and library realizations, not agent-specific system classes.

## Goals / Non-Goals

**Goals:**
- Implement the six models on existing primitives, with tests of their known behaviour and curated examples.

**Non-Goals:**
- New core primitives. A model that turns out to need one is blocked on a change that specifies it.

## Decisions

Each model follows its source section and the patterns of the implemented agent economies, such as `sugarscape` and
`double_auction`, as migrated by `redesign-system-model`:

| Need | Written with |
|---|---|
| ordered trading steps | `DiscreteTime(schedule=Phases([...]))` |
| agents, entry and exit | an `Element` with `Rule` laws in a `Group(variable=True)` |
| grid or road space | `LatticeSpace` or `NetworkSpace` with a neighbourhood and capacity |
| matching buyers and sellers, riders and drivers | `Match` with a pairing realization from `skcomplex.models.economics.realizations`, such as `OrderBook` or `BilateralSearch` |
| exchange of goods and money | `Transfer` laws, with an `Invariant` on each total |
| a regulator of the commons | `parameter(adaptive=True)` changed by a rule |
| recorded quantities | observables declared by the catalog entry |

`ridesharing` and `tragedy_of_the_commons` live in `skcomplex.models.social`; a pairing or learning realization that
only economics shares is defined again in the social package, as the realization rule requires. The models run on the
solver and backend that validation chooses for rule laws with variable groups, the Mesa backend by default.

The order follows the inventory, since all six have the same priority.

## Risks / Trade-offs

- [A repeated pattern hides a missing primitive] → The implementation stops and specifies it rather than working
  around it.
