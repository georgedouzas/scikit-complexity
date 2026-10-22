# Spec Delta

## ADDED Requirements

### Requirement: Evolutionary Sherrington-Kirkpatrick game
The library SHALL provide `evolutionary_sk_game` in `physics/criticality`, after Lesmana et al. (2026, arXiv
2604.15669): learning agents on a Sherrington-Kirkpatrick landscape with Bak-Sneppen replacement of the least fit,
built from the core's group, reduction, draw, and update primitives under discrete time, with the extremal replacement
an update switched by the group minimum, and with its softmax choice defined in its model package rather than imported
from another field's realizations.

#### Scenario: Rules of the game
- **WHEN** a user reads the factory of `evolutionary_sk_game`
- **THEN** couplings are symmetric Gaussian with variance `1/N`, each agent's incentive is updated from its local field
  with a memory weight, actions are softmax draws at the agent's temperature, and the least fit agent is replaced with a
  temperature redrawn uniformly on [0, 2]

#### Scenario: Self-organisation to the edge of ergodicity breaking
- **WHEN** `evolutionary_sk_game` runs with 256 agents and memory 0.75 past its transient
- **THEN** the replacement cascades follow a power law with exponent within 0.15 of 3/2, and the mean reward exceeds the
  reward of the same game run at every fixed temperature tried in a sweep

#### Scenario: Example of the evolutionary game
- **WHEN** a user opens `docs/examples/physics/criticality/plot_evolutionary_sk_game.py`
- **THEN** its thumbnail shows the evolved temperature distribution and its analysis compares the cascade sizes and the
  ergodicity-breaking diagnostic with the paper's results

### Requirement: Emergence models are recorded in the model inventory
The inventory SHALL record `evolutionary_sk_game` with its arXiv source, its primitives, and its priority, pending this
change until it is implemented.

#### Scenario: Entry of the evolutionary game
- **WHEN** a contributor reads the inventory entry of `evolutionary_sk_game`
- **THEN** it cites arXiv 2604.15669, is tagged with discrete time, no space, a group, no binding, and the discrete
  law class, and has priority `core`
