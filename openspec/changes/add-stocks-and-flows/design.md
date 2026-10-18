# Design

## Context

Rate, jump, and update laws target one state each, and contributions add up. Agents change their states one at a time
through rules. Observables, after `add-general-model-interface`, give named expressions per frame.

## Goals / Non-Goals

**Goals:**

- One flow primitive for system dynamics, mass balance, conservation laws, and stock-flow consistent economics.
- Feedback structure that users can inspect, as Meadows (2008) and Bala et al. (2017 ch. 3) teach.

**Non-Goals:**

- A graphical stock-and-flow editor. Loops and flows export to NetworkX and Graphviz for drawing.
- Physical units and dimensional analysis (Bala et al. 2017 ch. 6). Quantity tags leave room for units.

## Decisions

### Flows and quantities

`state(..., quantity='money')` tags a state. `Flow(source, sink, amount, label=None)` is a law. `source` and `sink` are
state symbols of the same quantity, or `Source()` and `Sink()` for the environment, and `amount` is a rate, a jump
size, or a per-step amount, depending on the system's form. The compiler expands a flow into one negative and one
positive contribution, and records the flow under its label so trajectories and Rerun show it like an observable.

### Conservation

For every quantity, `conservation(system)` checks symbolically that internal flows cancel in the sum of the tagged
states. A quantity whose total changes outside declared flows, for example through a plain rate law on a tagged state,
raises a validation error at compile time. In checked runs, a monitor compares the total's change with the integrated
boundary flows each frame and raises a named error on drift beyond the tolerance. `flow_matrix(trajectory, frame)`
returns the transaction-flow matrix, with flows as rows and holders as columns, whose rows sum to zero.

### Auxiliaries, lookups, and delays

`Auxiliary(label, expression)` binds a name that later laws use and that trajectories record. `lookup(x, points)` is a
Sage function of piecewise-linear interpolation with flat extrapolation, lowered to `np.interp` and its Numba
equivalent. `delay1`, `delay3`, and `smooth`, `smooth3` add hidden stocks per stage, as the system-dynamics literature
defines them, with the delay time as a parameter.

### Loop analysis

`loops(system, at=None)` builds a directed graph with an edge from each symbol to every state or auxiliary whose law
uses it. Each edge's sign is the sign of the partial derivative, derived with Sage and evaluated at a state when it
depends on one. Cycles are found with Johnson's algorithm, and each loop's polarity is the product of its edge signs.
`loop_dominance(trajectory)` ranks loops over time by the magnitude of their loop gains along the run.

### Ledgers

`Transfer(quantity, from_agent, to_agent, amount)` in agent rules moves an amount atomically between two agents' tagged
states. `Identity(expression)` declared on an agent type is checked after every step. Removing an agent requires a
settlement rule that moves its holdings, otherwise conservation fails with a named error.

## Risks / Trade-offs

- [Symbolic signs depend on the state] → Signs are evaluated at the given state or along the trajectory, and edges with
  changing signs are flagged.
- [Strict conservation rejects valid shortcuts] → Quantities are opt-in, and untagged states behave as today.
