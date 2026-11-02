# stocks-and-flows Specification

## Purpose
Lets a user build models from stocks and named flows of conserved quantities, with auxiliaries, lookup tables, delays,
feedback loop analysis, and balance-sheet ledgers for agents.

## Requirements

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

### Requirement: Feedback loops are derived from the laws
The analysis layer SHALL derive the signed dependency graph of a system, list its feedback loops with their polarity,
and rank loop dominance over a trajectory.

#### Scenario: Population loops
- **WHEN** a user lists the loops of a population with births and deaths proportional to its size
- **THEN** the result holds one reinforcing loop through births and one balancing loop through deaths
