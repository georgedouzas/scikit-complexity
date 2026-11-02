## ADDED Requirements

### Requirement: One partner-selection spec
Transfers, matching, and groupings SHALL accept one partner-selection spec made of candidates (all, neighbours, a radius,
or a radius read from a state), a choice among them (uniform, by score, nearest), a fallback when there is none (none or
the nearest partner), and a capacity, or a pairing callable instead.

#### Scenario: Within reach, else the nearest
- **WHEN** agents pick a provider within the radius in their `reach` state, falling back to the nearest provider
- **THEN** an agent with providers in reach picks one of them uniformly, and an agent with none picks the nearest

#### Scenario: Continuous space
- **WHEN** the candidates are a radius and the agents live in a continuous space
- **THEN** candidates are the agents within that Euclidean distance, with no grid required

#### Scenario: Pairing callable
- **WHEN** a matching receives a pairing callable instead of a spec
- **THEN** it pairs agents as the callable returns and applies its settlement to each pair

### Requirement: Settlement from one draw credits both ends
A transfer between two partners SHALL be able to credit and debit both ends from one shared draw.

#### Scenario: Meal
- **WHEN** an eater buys a meal whose calories are drawn once
- **THEN** the eater gains those calories and the provider's sales rise by one in the same step
