# Design

## Context

Rate, jump, and update laws target one state each, and contributions add up. Agents change their states one at a time
through rules. Observables give named expressions per frame, and the compiler already supports hidden entries, as
for delayed states and reductions.

## Goals / Non-Goals

**Goals:**

- One flow primitive for system dynamics, mass balance, conservation laws, and stock-flow consistent economics.
- Feedback structure that users can inspect, as Meadows (2008) and Bala et al. (2017 ch. 3) teach.

**Non-Goals:**

- A graphical stock-and-flow editor. The loop graph is a NetworkX graph that users can draw.
- Physical units and dimensional analysis (Bala et al. 2017 ch. 6). Quantity tags leave room for units.
- Delays in discrete maps and in groups. Delays and smoothings add hidden stocks to rate laws of single elements and
  interactions.

## Decisions

### Flows and quantities

`state(..., quantity='money')` tags a state. `Flow(source, sink, amount, label=None, rate=None)` is a law. `source` and
`sink` are state symbols of the same quantity, or `Source()` and `Sink()` for the environment, each with an optional
label for transaction-flow matrices. In a system of rate laws the amount is a rate, and the compiler expands the flow
into one negative and one positive rate law. With `rate`, the flow is a jump of size `amount` with that propensity. In a
system with update laws, the amount is moved once per step: the compiler adds the net flow of every state to the update
law of that state, or creates `x + net` when no update law sets it. The trajectory records every flow under
`<component path>.<label>`, which the observable evaluator computes like an observable.

The expansion runs in a new compiler pass, `compile/_flows.py`, called when laws are collected and once more after all
components are collected. It keeps the compiler's other passes unchanged: flows become ordinary rate, jump, and update
laws before the form is inferred.

### Conservation

The compile-time check raises `ConservationError`, a `ValidationError`, for any rate, noise, jump, or update law outside
flows that changes a tagged state, naming the state and its quantity. `conservation(system)` sums the flows of every
quantity, checks symbolically that internal flows cancel, and returns the net inflow from the environment. For every
quantity with boundary flows the compiler adds a hidden state `__boundary.<quantity>` that accumulates that inflow.
`ConservationMonitor`, an observer, compares at every frame the change of each total with its accumulated inflow and
raises `ConservationError` on drift beyond its tolerance, for example after a boundary law clips a stock.
`flow_matrix(trajectory, frame)` returns the transaction-flow matrix, with flows as rows and holders, the states and the
environments, as columns, whose rows sum to zero.

### Auxiliaries, lookups, and delays

`Auxiliary(label, expression)` binds a name, and laws use `auxiliary.symbol`, which the pass substitutes. Trajectories
record the auxiliary, and the loop analysis keeps it as a node. `lookup(x, points)` reuses `series`, the nested
conditions of piecewise-linear interpolation with flat extrapolation that every backend already lowers, so no new
printer is needed. `delay1`, `delay3`, `smooth`, and `smooth3` are Sage functions that the pass replaces by hidden stocks
`<component path>.<kind>_<index>`, one per stage, with the delay time as an expression of parameters: a material delay
of order `n` drains each stage at `stage / (time / n)` and outputs the last drain, and a smoothing of order `n` chains
first-order adjustments. Hidden stocks start in equilibrium with the input's initial value.

### Loop analysis

`loops(system, at=None)` builds a NetworkX directed graph with an edge from each state, auxiliary, or flow to every
auxiliary, flow, or state rate whose expression reads it, and from each flow to its sink (`+`) and source (`-`). Each
edge's sign is the sign of the partial derivative, derived with Sage and evaluated at the given state or the initial
state. Delays and smoothings pass their input with gain one. Cycles are found with NetworkX's `simple_cycles`, Johnson's
algorithm, and each loop's polarity is the product of its edge signs. `loop_dominance(trajectory)` evaluates the edge
derivatives over all frames at once and ranks loops by the magnitude of their gains, the products of the partial
derivatives along them. Loops take systems of single elements.

### Ledgers

`transfer(source, sink, amount, state, sink_state=None)` in agent rules moves an amount between two agents' states at
once. Debts are negative holdings, so a loan is a transfer of claims and total holdings of each instrument stay zero.
`Identity(label, residual, tolerance=...)` objects listed in the class variable `identities` of an agent type are
checked by the agent stepper after every step, and a violation raises `ConservationError` naming the agent, the
identity, and the difference. Removing an agent that still holds a nonzero tagged state raises `ConservationError`, so
a settlement rule must move its holdings first.

### Library

The surveyed system-dynamics models are registered under their inventory names, and `bathtub` and `car_dealer` are
rebuilt on flows under the same names. The existing `cooling_coffee` and `thermostat` models stay as they are, since they
hold no conserved quantity moved between holders. The monetary economy is a discrete circuit of deposits between
households, firms, and a bank, in which lending and repayment are the only flows with the environment.

### Libraries surveyed

- NetworkX provides Johnson's cycle enumeration and the graph type, and is already a dependency.
- PySD and BPTK-Py translate and simulate Vensim and XMILE models with their own integrators. Taking them would replace
  the compiler and its backends rather than lower flows to them, so they were rejected; the flow pass instead lowers to
  the existing rate, jump, and update laws.

## Risks / Trade-offs

- [Symbolic signs depend on the state] → Signs are evaluated at the given state or along the trajectory.
- [Strict conservation rejects valid shortcuts] → Quantities are opt-in, and untagged states behave as today.
- [Partitioned problems have no stock-flow description] → The analysis functions and the monitor raise a named error
  for them.
