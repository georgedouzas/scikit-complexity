# stocks-and-flows Specification

## Purpose
Lets a user build models from stocks and named flows of conserved quantities, with auxiliaries, lookup tables, delays,
feedback loop analysis, and balance-sheet ledgers for agents.

## Requirements

### Requirement: Flows move quantities between holders
A flow SHALL move an amount of a quantity from a source state to a sink state, or exchange it with the environment,
and SHALL appear in trajectories under its label.

#### Scenario: Bathtub
- **WHEN** a bathtub stock has an inflow of 5 and an outflow proportional to its level
- **THEN** the stock approaches the level where the flows balance, and the trajectory records both flows

### Requirement: Conserved quantities are checked
Every quantity SHALL change only through declared flows, which the compiler checks symbolically and checked runs
monitor against boundary flows.

#### Scenario: Undeclared change
- **WHEN** a rate law changes a state tagged `money` outside any flow
- **THEN** compiling raises a named validation error that names the state and the quantity

#### Scenario: Closed economy
- **WHEN** a closed economy of households, firms, and a bank runs for 100 steps
- **THEN** the total money changes only by the bank's declared lending and repayment, and every row of the
  transaction-flow matrix sums to zero

### Requirement: Auxiliaries and lookups shape laws
Laws SHALL use named auxiliaries and piecewise-linear lookup tables, which trajectories record and every backend lowers.

#### Scenario: Table function
- **WHEN** a fertility auxiliary uses a lookup table of food per person
- **THEN** its values interpolate linearly between the table's points and stay flat outside them

### Requirement: Delays and smoothing add hidden stocks
Laws SHALL use first- and third-order material delays and information smoothing with a delay time as a parameter.

#### Scenario: Third-order delay response
- **WHEN** a step input enters a third-order delay with delay time 6
- **THEN** the output follows the Erlang-3 step response and reaches half the step near time 5.35

### Requirement: Feedback loops are derived from the laws
The analysis layer SHALL derive the signed dependency graph of a system, list its feedback loops with their polarity,
and rank loop dominance over a trajectory.

#### Scenario: Population loops
- **WHEN** a user lists the loops of a population with births and deaths proportional to its size
- **THEN** the result holds one reinforcing loop through births and one balancing loop through deaths

### Requirement: Ledgers keep agents' balance sheets consistent
Agent rules SHALL move quantities between agents with atomic transfers, identities declared on agent types SHALL hold
after every step, and removed agents SHALL settle their holdings.

#### Scenario: Identity violation
- **WHEN** a rule lowers a bank's assets without lowering its liabilities or equity
- **THEN** the step raises a named error that names the bank, the identity, and the difference

#### Scenario: Bankruptcy settlement
- **WHEN** a bankrupt firm is removed
- **THEN** its debt is written off against its lenders through transfers, and total money stays consistent
