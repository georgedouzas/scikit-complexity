# Proposal

## Why

Agent-based economics and social models share structure that the framework leaves to hand-written rules:

- **Ordered phases:** Caiani et al. (2016 §2.2) list fifteen ordered events per period, and run a stock market with
  daily and intraday loops (ch. 3). Chesney (2021 ch. 1) separates offer, acceptance, and payment phases.
- **Market protocols:** firms sampling banks and switching lenders (Caiani et al. 2016 ch. 2), a limit order book (ch.
  3), a Walrasian auctioneer against bilateral barter (Hamill and Gilbert 2016 ch. 5), job matching (ch. 7), and double
  auctions (Gooding 2019 ch. 11).
- **Institutions:** a central bank, a regulator, a minimum wage, or a grazing quota reads aggregates and sets
  parameters for everyone else (Hamill and Gilbert 2016 ch. 9-10; Chesney 2021 ch. 3; Caiani et al. 2016 §1.2.6).
- **Learning and evolution:** expectation heuristics with logit switching (Caiani et al. 2016 §1.2.2), imitation of
  successful neighbours (Argun et al. 2021 ch. 13), and genetic algorithms (Gooding 2019 ch. 6).

## What Changes

- Add phases: a step of an agent partition runs named phases in order. Each phase has its own rules and an activation
  policy: random order, sequential, simultaneous with a two-buffer update, or by agent type. Phases can repeat as
  sub-steps, such as intraday rounds within a day.
- Add market protocols: interactions between two agent groups that match participants and clear trades. Protocols
  include random bilateral search, sampling and switching, a rationed queue, a limit order book with price-time
  priority, a call auction, a Walrasian auctioneer, and deferred acceptance through the `matching` library. They report
  prices and volumes as reporters, and settle trades by atomic transfers of money and goods between the agents, which
  can move to the ledger transfers of `add-stocks-and-flows` once that change lands.
- Add institutions: singleton agents whose rules read observables and set parameters of other groups, live during the
  run.
- Add learning blocks:
  - the agent's own past values over a window;
  - forecasting heuristics (adaptive, trend-following, anchoring);
  - logit switching among strategies by past performance;
  - imitation of the best neighbour;
  - selection, crossover, and mutation over agent strategies.
- Add library models that probe these blocks in distinct ways: El Farol bar, the minority game, the Kolkata restaurant
  game, kinetic wealth exchange, Brock-Hommes heuristic switching, a double auction, fractional-reserve banking, a
  firm-bank credit network, a job-search labour market, fair-food choice, Sugarscape, and Axelrod culture
  dissemination. The existing spatial prisoner's dilemma covers spatial cooperation. Models that only repeat covered
  patterns (goods traders, shops, barter, an order-book market with imitation, the meadow commons, and ridesharing)
  stay pending this change.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `agent-based-modeling`: phases and activation policies, market protocols, institutions, and learning blocks.
- `model-library`: economic and social agent models.

## Impact

- `skcomplex.core.model` gains `Phase`, `Market` with its protocols, `Institution`, `memory`, `heuristics`,
  `logit_probabilities`, `logit_choice`, `imitate`, and `evolve`. `System` gains `phases`, `Rule` gains `phase`, and
  `AgentType` gains the class variable `remembers`.
- The Mesa stepper runs phases when a system declares them, and keeps one activation per step otherwise. Rules gain
  `model.report`, `model.history`, `model.storage`, `model.round`, `model.observable`, and `model.set_parameter`.
- Adds the `matching` library as a runtime dependency.
- Settles trades without ledgers, so it does not wait for `add-stocks-and-flows`.
