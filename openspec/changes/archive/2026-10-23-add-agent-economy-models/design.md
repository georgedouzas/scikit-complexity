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

### Decisions made during implementation

- The inventory files `tragedy_of_the_commons` under `economics/resources` and `ridesharing` under
  `engineering/traffic`, and the inventory check requires a model to register under its entry's field path. The models
  follow the inventory, not the `social` placement above. `tragedy_of_the_commons` pairs cows with pastures by its own
  `RichestPasture`, and `ridesharing` pairs riders with cabs by its own `NearestCab`, since no other package shares
  them.
- The total of a good over a group of agents had no invariant, since an agent invariant was a function of one member.
  The core `Invariant` now also takes, when an interaction of agents declares it, a module-level function of the
  interaction's model view, which the agent stepper checks after every step. This widens an existing primitive and
  adds none. `goods_traders`, `barter`, and `order_book` declare an invariant on the total of every traded holding.
- Agent states are not symbols, so the exchanges inside agent rules use the imperative `transfer`, the agent form of
  `Transfer`. The law census sees matching, rule, and invariant laws, so the inventory tags record those, not
  `transfer`.
- `Law` does not include `Match` or `Invariant`, so a `laws` method that returns them is typed `list[Any]`.
- The bilateral Cobb-Douglas barter of `goods_traders` and `barter` is the realization `trade_goods`, with
  `cobb_douglas`, `marginal_rate`, and `NeighbourPairing`, in `skcomplex.models.economics.realizations`.
- `shops` first used `DeferredAcceptance`, whose `matching` solver took about a second a step for 400 shoppers, beyond
  the gallery budget. The shoppers instead choose in random order the shop of the lowest price plus travel cost, the
  pairing `DeliveredPriceChoice`, which is the stable matching when shops do not rank shoppers.
- Interactions hold no adaptive parameter, so the regulated quota of `tragedy_of_the_commons` is the adaptive
  parameter `quota` of every herder, which the regulator changes.
- The Edgeworth box has no space, so `barter` runs without one and records the primitives `agents`, `phases`, and
  `markets`. By the inventory rule, `goods_traders` and `barter` are `core`, the first implemented entries of their
  primitive sets, and the other four are `example`.

## Risks / Trade-offs

- [A repeated pattern hides a missing primitive] → The implementation stops and specifies it rather than working
  around it.
