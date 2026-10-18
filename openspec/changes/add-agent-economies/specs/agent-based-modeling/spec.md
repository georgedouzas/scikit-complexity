# Spec Delta

## ADDED Requirements

### Requirement: Steps run ordered phases
An agent partition SHALL run named phases in order every step, each with its own rules, an activation policy (random,
sequential, simultaneous, or by type), and an optional number of repeated rounds.

#### Scenario: Offer before acceptance
- **WHEN** employers make wage offers in one phase and workers accept the best offer in the next
- **THEN** every acceptance in a step refers to an offer made earlier in the same step

#### Scenario: Simultaneous imitation
- **WHEN** agents on a lattice copy their best neighbour's strategy in a simultaneous phase
- **THEN** every agent reads its neighbours' strategies from the start of the phase

### Requirement: Markets match and clear between groups
A market interaction SHALL match two agent groups with a chosen protocol, settle trades through ledger transfers, and
publish price, volume, and unmatched quantity as observables.

#### Scenario: Double auction efficiency
- **WHEN** zero-intelligence traders with budget constraints trade in a call auction
- **THEN** the realised surplus exceeds 90% of the maximum surplus implied by their valuations

#### Scenario: Order book priority
- **WHEN** two sell orders at the same price arrive before a matching buy order
- **THEN** the earlier sell order trades first

### Requirement: Institutions set parameters from observables
An institution SHALL read observables and set parameters of other groups and interactions during the run, with every
change recorded.

#### Scenario: Interest-rate rule
- **WHEN** a central bank raises the interest rate whenever inflation exceeds its target
- **THEN** the interest rate in the trajectory changes only at steps where the inflation observable exceeds the target

### Requirement: Learning blocks support adaptation
Agent rules SHALL use memories of their own states, forecasting heuristics, logit choice among strategies, imitation of
neighbours, and evolutionary operators over strategy states.

#### Scenario: El Farol self-organisation
- **WHEN** 100 agents with heuristic forecasts decide whether to attend a bar with capacity 60 for 500 steps
- **THEN** the mean attendance after the transient lies within 5 of 60
