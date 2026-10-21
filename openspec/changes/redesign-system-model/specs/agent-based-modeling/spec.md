# Spec Delta

## MODIFIED Requirements

### Requirement: Agent systems run on Mesa
An agent system SHALL be an ordinary system: elements with rule laws, in fixed or variable groups, under discrete time,
on a space. Mesa SHALL be the backend that executes discrete-time systems with rule laws, selected and validated like
any other backend. It SHALL build a Mesa model from the system, advance one Mesa step per unit of time, and activate
agents by the schedule of the system's time, with a seed that makes runs reproducible. Each step SHALL advance Mesa's
clock, so the model time that rules read during the step that produces frame `k` is `k`.

#### Scenario: Reproducible run
- **WHEN** the same agent system is simulated twice on the Mesa backend with random state 0
- **THEN** the two trajectories are equal

#### Scenario: Rules read the model time
- **WHEN** a rule stores `model.time` in a state of its agent at every step of a run until time 3
- **THEN** the state is 1, 2, and 3 in the frames at times 1, 2, and 3, not 0

#### Scenario: No agent-specific system class
- **WHEN** a user builds an agent model
- **THEN** they use `System` with `DiscreteTime`, and no `MesaSystem` or `AgentType` exists in the core

### Requirement: Steps run ordered phases
A discrete time with a phases schedule SHALL run its named phases in order in every step. Each phase SHALL have its
own rules, an activation policy (random, sequential, simultaneous, or by type), and an optional number of repeated
rounds. Rules without a phase SHALL run in the phase `'default'`, first unless a phase of that name is listed.

#### Scenario: Offer before acceptance
- **WHEN** employers make wage offers in one phase and workers accept the best offer in the next
- **THEN** every acceptance in a step refers to an offer made earlier in the same step

#### Scenario: Simultaneous imitation
- **WHEN** agents on a lattice copy their best neighbour's strategy in a simultaneous phase
- **THEN** every agent reads its neighbours' strategies from the start of the phase

#### Scenario: Phases as a time schedule
- **WHEN** a system declares `DiscreteTime(schedule=Phases([produce, trade, consume]))`
- **THEN** every step runs the rules of `produce`, then `trade`, then `consume`

## REMOVED Requirements

### Requirement: Existing Mesa models run unchanged
**Reason**: Mesa is an execution backend, not a kind of system, and a wrapped Mesa model has no components,
interactions, or space the framework can validate or analyse.
**Migration**: Port the model to a system with rule laws and run it on the Mesa backend.

### Requirement: Markets match and clear between groups
**Reason**: A market is a field-specific realization. Its general primitives are matching, transfer, invariants, and
algebraic clearing of adaptive parameters.
**Migration**: Use the market building blocks of `skcomplex.models.economics`, built on `Match`, `Transfer`,
`Invariant`, and `Equation`.

### Requirement: Institutions set parameters from observables
**Reason**: An institution is a field-specific realization of adaptive parameters changed by laws.
**Migration**: Declare the parameter `adaptive=True` and change it with a rule or update law of the component that
governs it.

### Requirement: Learning blocks support adaptation
**Reason**: Forecasting heuristics, logit choice, imitation, and evolutionary operators are specific realizations of
draws, lags, and reductions.
**Migration**: Import them from the learning helpers of `skcomplex.models.economics` or `skcomplex.models.social`.
