# Design

## Context

Agent partitions run on Mesa with one rule pass per step in random order. Agents see their neighbours, the model's
random stream, other agents by group, and inputs from other partitions. Ledgers, after `add-stocks-and-flows`, move
quantities between agents atomically.

## Goals / Non-Goals

**Goals:**

- Economic models built as configurations of phases, protocols, institutions, and learning blocks, so the library's
  economic models are short factories.
- Protocols reusable outside economics, for example for mate choice, job allocation, or resource queues.

**Non-Goals:**

- A full macroeconomic benchmark model. The firm-bank credit network of Caiani et al. (2016 ch. 2) is the largest model
  in scope.
- Discrete-event scheduling with continuous event times for queues (Chesney 2021 ch. 6). Jump laws cover simple queues
  until a later change.

## Decisions

### Phases

`AgentType.phases` or `System(phases=[Phase(name, rules, activation='random' | 'sequential' | 'simultaneous' |
'by-type', repeat=1)])` declares the order. Rules from `laws` without a phase run in a default phase, so existing
models are unchanged. Simultaneous activation gives every rule read access to the states at the start of the phase and
applies writes at its end. Repeats run inner rounds, and observables can be recorded per round.

### Market protocols

`Market(buyers, sellers, protocol, quantity, price, label)` is an interaction between two groups. The protocol objects
are:
- `BilateralSearch(tries)`;
- `SampleAndSwitch(sample, intensity)`, which switches partners with probability `1 - exp(intensity * Δ / reference)`;
- `RationedQueue(order)`;
- `OrderBook()`;
- `CallAuction()`;
- `Walrasian(adjustment)`;
- `DeferredAcceptance()`.

Agents express demand and supply through declared attributes or functions. The market runs in its phase, settles every
trade with ledger transfers, and publishes `price`, `volume`, and `unmatched` as observables.

### Institutions

`Institution(label, rules)` is an agent type with one member. Its rules read observables through the model view, and
set parameters of groups and interactions through `model.set_parameter(path, value)`. That call uses the live parameter
mechanism and is recorded in the trajectory.

### Learning blocks

- `memory(agent, state, window)` returns the agent's last values. The stepper records them only for states that rules
  declare in `remembers`.
- `heuristics.adaptive`, `.trend`, and `.anchoring` give forecasts from memories.
- `logit_choice(utilities, intensity, rng)` chooses a strategy by past performance.
- `imitate(agent, neighbors, score)` copies the best neighbour.
- `evolve(group, fitness, select, crossover, mutate)` applies a genetic algorithm to strategy states, with the model's
  random stream.

## Risks / Trade-offs

- [Phases change performance] → Phases reuse Mesa's agent sets, and simultaneous activation copies only the states that
  phases write.
- [Protocol semantics vary in the literature] → Each protocol documents its rules and source, and tests check its
  clearing against small worked examples.
