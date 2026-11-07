# Design

## Context

Economics systems are built from elements with symbolic laws (maps, rates, transfers) or from groups of passive agents
that an interaction with a rule updates, as `debtrank` does. Discrete maps read lagged values with `lag` and draw with
`normal`, `uniform`, and `bernoulli`, and `TransitionLaw` steps a finite Markov chain. See proposal.md for the list.

## Goals / Non-Goals

**Goals:**
- Follow the rules, parameters, and defaults of each source, and say in the docstring where the source is a
  narrative and the rules are a reconstruction.
- Keep default runs to a few seconds.

**Non-Goals:**
- Core changes. A model that needs one is reported instead.
- Calibrations beyond the published defaults.

## Decisions

- **Maps for STELLA and discrete-time models.** The crop livestock household uses `INT` and a pipeline `DELAY` of
  three years with `dt = 1`, so it is an update map whose purchases enter the herd through lags. The rice food balance
  has no discontinuity of that kind and is a continuous stock-and-flow system like `food_security`.
- **Measurement errors as a stochastic map.** Argun's fishery is an SDE whose quota uses a measurement taken at every
  step, so it is an Euler-Maruyama map with named normal draws for the growth noise and the error.
- **Markets of passive agents updated by one feedback interaction.** The guru order book, the Fair Food Program, the
  default cascade, Eisenberg-Noe, the fire sales, the CCC model, El Farol coalitions, Mark-0, the Toy Trader, and the
  Gooding double auction have sequential or global steps (an order book within a day, buyers that fill orders farm by
  farm, traders that buy one after another and then all move, synchronous balance-sheet updates). One
  `FeedbackInteraction` per system acts once per step on the groups, as `HeuristicPricingInteraction` does, and the
  agents stay groups whose states the trajectory records. The Toy Trader keeps its places as states on a torus and
  finds the sellers within its market sight itself, which is faster than Mesa radius queries for hundreds of traders.
- **One system per model family.** The threshold cascade of Thurner and the Gai-Kapadia cascade differ only by the
  threshold and the recovery rate, so `default_cascade` takes both. The Cobb-Douglas and the elasticity demand of
  Hamill share households and budgets, so `household_demand` takes `demand`. The three-goods trader is `n_goods=3` of
  `toy_trader`.
- **Decoupling as an internal model.** Forrest's feedback component holds a copy of the plant state and the states
  of the decoupled systems. The copy equals the plant state from the shared zero start, so the controller reads the
  plant state and keeps the decoupled states.

- **No separate urn with triggering.** `interacting_urns_network` on a graph of one node, or on isolated nodes, is the
  urn with triggering of Tria et al. that 2012.06369 reviews, so the audited `urn_with_triggering` is not added.

## Risks / Trade-offs

- [Narrative sources] → the rules follow the text and the docstrings say which choices are reconstructions.
- [Slow agent runs] → default sizes and horizons are reduced, and the source sizes are arguments.
- [Defaults that differ from the source] → the Kirman market uses 30 traders so that regimes switch within a run, and
  the CCC model uses `r+ = 5`, `r- = 10`, since `r+ = 10`, `r- = 15` restructures almost every step with these
  tensors. The docstrings say so.
