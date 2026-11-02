# Proposal

## Why

Systems science describes systems as stocks connected by flows, under feedback loops, with conserved quantities moving
between holders. Sources:
- **System dynamics:** Bala et al. (2017 ch. 3-4) and Meadows (2008) build models from stocks, flows, auxiliaries, table
  functions, and delays, and reason about reinforcing and balancing loops.
- **Systems science:** Mobus (2022 §4.3.3) types every flow as material, energy, or message, between components and
  environmental sources and sinks.
- **Stock-flow consistency:** economic models must be stock-flow consistent, so that every transfer is one holder's
  outflow and another's inflow and balance sheets obey their identities. Gooding (2019 ch. 13-14) monitors that debits
  minus credits equal zero. Hamill and Gilbert (2016 ch. 9) and Caiani et al. (2016 ch. 2) keep bank and firm balance
  sheets.
- **Conservation in physics and chemistry:** the same flow abstraction gives mass balance in ecology and conservation
  laws in physics and chemistry (Schwichtenberg 2020 ch. 1-2 on a conserved good on a lattice).

Today flows are anonymous terms inside rate laws, conservation is only a convention, and agents change balances one at a
time.

## What Changes

- Add named flows: one flow moves an amount of a quantity from a source holder to a sink holder. It compiles to
  matching outflow and inflow terms for rate, jump, and update laws.
- Add sources and sinks for exchanges with the environment.
- Add quantity tags on states, such as `money`, `mass`, or `energy`. Every declared flow conserves its quantity.
- Add conservation checks: a symbolic check that internal flows cancel, a run-time monitor of each quantity's total
  against its boundary flows, and a transaction-flow matrix per step.
- Add auxiliaries: named intermediate expressions that laws reuse and that trajectories record.
- Add lookup tables: piecewise-linear functions of one variable that every backend lowers.
- Add system-dynamics delays and smoothing: first- and third-order material delays and information smoothing, built as
  hidden stocks.
- Add loop analysis: the signed dependency graph of states and auxiliaries derived from the laws, its feedback loops
  with polarity, and the dominant loop over time by loop gains.
- Add ledgers for agents: atomic transfers between agents, identities such as `assets == liabilities + equity` checked
  every step, and settlement of holdings when an agent is removed.
- Add library models:
  - Meadows' systems zoo: bathtub, coffee cooling, thermostat, inventory with delays, renewable and nonrenewable
    resources.
  - Meadows' system traps: success to the successful, shifting the burden, escalation.
  - Bass diffusion and the beer game.
  - A stock-flow consistent monetary economy with households, firms, and a bank.

## Capabilities

### New Capabilities

- `stocks-and-flows`: flows, quantities, conservation, auxiliaries, lookups, delays, loop analysis, and ledgers.

### Modified Capabilities

- `model-library`: system-dynamics and stock-flow consistent models.

## Impact

- `skcomplex.core.base` gains the `quantity` argument of `state` and `ConservationError`.
- `skcomplex.core.model` gains `Flow`, `Source`, `Sink`, `Auxiliary`, `lookup`, `delay1`, `delay3`, `smooth`,
  `smooth3`, `transfer`, and `Identity`, and agent types gain the class variable `identities`.
- `skcomplex.core.analysis` gains `loops`, `loop_dominance`, `conservation`, and `flow_matrix`, and
  `skcomplex.core.runtime` gains `ConservationMonitor`.
- The compiler gains a flow pass in `compile/_flows.py`, and `Problem` gains `stock_flow`.
