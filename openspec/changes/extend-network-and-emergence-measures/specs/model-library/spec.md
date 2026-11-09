## ADDED Requirements

### Requirement: Surviving by serving
The library SHALL provide `SurvivingByServingSystem` in `biology/evolution` (Metzner et al. 2026, arXiv 2606.26733):
per episode a source adds two raw unit vectors to a shared pool, every pooled vector ages out with a probability,
agents in random order transform two matching vectors by cosine threshold into two copies of an output, and two
evaluators take matching vectors. Credited agents reset their uncredited count, and others mutate by a Hill-type
probability.

#### Scenario: Pool without agents
- **WHEN** the system runs with no agents and no evaluators for 20000 episodes at an ageing probability `P`
- **THEN** the mean pool size at the end of an episode is `2 (1 - P) / P` within 5%

#### Scenario: Exclusion rules
- **WHEN** the system runs with its defaults
- **THEN** no agent consumes its own output, no two inputs of one transformation come from the same agent, and two raw
  inputs are always of different types

#### Scenario: Credit balance
- **WHEN** the system runs with its defaults
- **THEN** the total credit equals the number of consumed agent outputs plus the number of evaluator selections, and
  the recorded transfers between agents sum to the number of consumed agent outputs

#### Scenario: Hill adaptation
- **WHEN** many agents without credit reach `T_ada` uncredited episodes
- **THEN** half of them mutate at that episode within the sampling error

#### Scenario: Recorded in the inventory
- **WHEN** the inventory check runs
- **THEN** the system registers as `surviving_by_serving` under `biology.evolution` with an implemented entry and a
  gallery example
