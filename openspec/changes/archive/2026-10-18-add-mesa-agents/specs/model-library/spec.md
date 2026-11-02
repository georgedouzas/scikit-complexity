# Spec Delta

## ADDED Requirements

### Requirement: Social agent-based models
The library SHALL provide a `social` module with Schelling segregation, Boltzmann wealth exchange, and the voter model.

#### Scenario: Schelling segregation increases
- **WHEN** a Schelling model with a similarity threshold of 0.5 runs until no agent moves or 100 steps pass
- **THEN** the share of similar neighbours is higher at the end than at the start

#### Scenario: Wealth stays constant
- **WHEN** a Boltzmann wealth model runs for any number of steps
- **THEN** the total wealth of all agents stays equal to its initial value

#### Scenario: Voter consensus on a small network
- **WHEN** a voter model runs on a connected network of 20 agents for long enough
- **THEN** every agent ends with the same opinion

### Requirement: Agent-based predator and prey
The ecology module SHALL provide a wolf and sheep model on a grid with grass regrowth, in which agents gain energy by
eating, reproduce, and die.

#### Scenario: Populations change
- **WHEN** the wolf and sheep model runs for 100 steps with its default parameters
- **THEN** both the wolf and sheep counts change over the run and are reported every frame
