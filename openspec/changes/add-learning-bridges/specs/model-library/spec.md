# Spec Delta

## ADDED Requirements

### Requirement: Cart and pole agrees with Gymnasium
The environment of `cart_pole` SHALL agree step by step with Gymnasium's `CartPole-v1`, the reference implementation of
the constants of Barto, Sutton, and Anderson (1983) and of the equations of Florian (2007).

#### Scenario: Same states for the same actions
- **WHEN** the environment of `CartPoleSystem` and `CartPole-v1` start from the same state and take the same 200 random
  actions
- **THEN** their observations agree within 1e-12 at every step until one terminates, and both terminate at the same
  step
