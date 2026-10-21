# Design

## Context

Agent partitions run on Mesa. A step has one rule pass, with agents in random order. Agents see their neighbours, the
model's random stream, other agents by group, and inputs from other partitions. Interactions with `Rule` laws run once
per step after the agents. `Observable`s of agent groups are evaluated at frames, and the stepper's snapshot carries
reporters such as `<group>.count` into the trajectory. Live parameters change through `Simulation.update_parameters`
and schedules, and every frame records the parameter vector.

Ledgers and their atomic transfers belong to `add-stocks-and-flows`, which is not merged yet. Markets therefore settle
trades with transfers of agent states, as described below.

## Goals / Non-Goals

**Goals:**

- Build economic models from phases, protocols, institutions, and learning blocks, so the library's economic models
  are short factories.
- Make protocols reusable outside economics, for example for mate choice, job allocation, or resource queues.
- Use mature libraries where they fit:
  - Mesa runs the agents.
  - The `matching` library (Wilde, Knight, and Gillard) solves deferred acceptance as a hospital-resident game.
  - SciPy's `softmax` computes logit probabilities.

**Non-Goals:**

- A full macroeconomic benchmark model. The firm-bank credit network of Caiani et al. (2016 ch. 2) is the largest model
  in scope.
- Discrete-event scheduling with continuous event times for queues (Chesney 2021 ch. 6). Jump laws cover simple queues
  until a later change.

## Decisions

### Phases

- `System(phases=[Phase(name, activation='random' | 'sequential' | 'simultaneous' | 'by-type', repeat=1,
  groups=None)])` declares the order of the parts of a step. `Rule(function, phase=name)` assigns a rule of an agent
  type, an interaction, or an institution to a phase.
- Rules without a phase run in the phase `'default'`. That phase runs first unless the system lists a phase of that
  name.
- A system that declares no phases and names none keeps the step it had before, so existing models are unchanged.
- In each round of a phase, the agents run their rules of the phase in the order the activation policy gives. The
  model rules of the phase then run once.
- Simultaneous activation buffers every state write of the phase and applies the writes at the end of the phase. Every
  rule therefore reads the states from the start of the phase, its own included.
- `model.round` gives the index of the current round.
- A system that names an undeclared phase is rejected when it starts.
- Phases are declared on the system rather than on agent types: one ordering covers agents, markets, and institutions
  together.

### Market protocols

- `Market(label, buyers=..., sellers=..., protocol=..., phase=...)` is an interaction between two groups bound to
  roles.
- Buyers state limit prices in `bid` and quantities in `demand`. Sellers state them in `ask` and `supply`. The names
  are arguments of the market, and a missing quantity counts as one unit.
- Each protocol receives the orders and returns the trades. The protocols are:
  - `BilateralSearch(tries, pricing)`;
  - `SampleAndSwitch(sample, intensity, partner)`, which switches partners with probability
    `1 - exp(intensity * (p_new - p_old) / p_new)`;
  - `RationedQueue(order)`;
  - `Walrasian(adjustment, iterations, tolerance)`, a tâtonnement followed by trade at one price;
  - `OrderBook(arrival)`, with price-time priority;
  - `CallAuction()`, at a uniform price;
  - `DeferredAcceptance(buyer_preference, seller_preference, pricing)`, which wraps `matching`.
- Each trade moves `price * quantity` of `money` from the buyer to the seller and `quantity` of `goods` from the seller
  to the buyer, and lowers `demand` and `supply`. Both sides are updated together, so money and goods are conserved. A
  buyer's money bounds what it buys.
- `on_trade(trade, model)` adds model-specific effects, such as recording an employer.
- When ledgers land, `settle` can move quantities through ledger transfers without changing the protocols.
- The market publishes `price`, `volume`, and `unmatched` as the reporters `<market>.price`, `<market>.volume`, and
  `<market>.unmatched`.

### Reporters, storage, and model access

- An interaction or agent type declares the names it reports in the class variable `reports`. `model.report(name,
  value)` publishes the reporter `<owner>.<name>`.
- Every declared reporter exists from the first frame, as `NaN` before its first report, so trajectories stay
  rectangular.
- `model.history(name, window)` returns the past reported values, one per report, so repeated rounds record one value
  per round.
- `model.storage` is a mapping private to the owner that lasts for one run. A clearing price or an order book's state
  can live there rather than on the component, which is cloned and pickled.

### Institutions

- `Institution(label, rule=None, phase=None)` is an interaction with one rule. Subclasses declare parameters and
  override `act(model)`.
- It reads observables with `model.observable(name)`, reporters with `model.history`, and agents with `model.agents`.
- It sets parameters of any group or interaction with `model.set_parameter(name, value)`. The stepper writes the new
  value into its own parameters and into the simulation's parameter vector, so rules see the change at once and the
  next frame records it.
- An institution is a single interaction rather than an agent type with one member: it needs no location or agent
  state, and it runs after the agents of its phase.

### Learning blocks

- `memory(agent, state, window)` returns an agent's last values of a state. The stepper records them at the end of
  every step, only for the states that the agent type lists in the class variable `remembers`, a mapping from state
  to the number of values kept.
- `heuristics.adaptive`, `.trend`, `.anchoring`, and `.fundamental` give the forecasts of Anufriev and Hommes from past
  values.
- `logit_probabilities(utilities, intensity)` and `logit_choice(utilities, intensity, rng)` choose by past performance.
- `imitate(agent, neighbors, score, states)` copies the states of the best-scoring neighbour.
- `evolve(genomes, fitness, rng, select, crossover, mutation, elite, ...)` applies selection, one-point crossover, and
  mutation to an array of strategies, using the model's random stream.
- DEAP was considered for `evolve` and rejected. Its operators draw from Python's global `random` module, which breaks
  the per-run seeding of simulations and experiments.

## Risks / Trade-offs

- [Phases change performance] → Phases reuse Mesa's agent sets. Simultaneous activation buffers only the states that
  rules write.
- [Protocol semantics vary in the literature] → Each protocol documents its rules and source, and tests check its
  clearing against small worked examples.
- [Transfers of agent states, not ledgers] → Settlement lives in one method of `Market`, which can be switched to
  ledger transfers when `add-stocks-and-flows` lands.
