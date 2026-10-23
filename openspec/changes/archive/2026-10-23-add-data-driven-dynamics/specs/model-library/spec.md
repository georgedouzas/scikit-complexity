# Spec Delta

## ADDED Requirements

### Requirement: Echo state network model
The library SHALL provide `echo_state_network` in `computer_science/neural_networks`, a reservoir of leaky tanh units driven by an input, with a curated gallery example.

#### Scenario: Echo state property
- **WHEN** `echo_state_network` runs from two initial states with spectral radius 0.9
- **THEN** the two reservoir trajectories converge, which is the echo state property
