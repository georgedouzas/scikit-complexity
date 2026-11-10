# Spec Delta

## ADDED Requirements

### Requirement: Learning prisoner's dilemma
The library SHALL provide `learning_prisoners_dilemma` in `social/cooperation`, two learners of the iterated prisoner's
dilemma with a memory of the last joint action and the payoffs of Foerster et al. (2018), with a gallery example and an
inventory entry.

#### Scenario: Payoffs of the paper
- **WHEN** the players always cooperate, then always defect
- **THEN** each player receives -1 per step, then -2 per step
