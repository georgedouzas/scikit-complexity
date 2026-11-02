# Spec Delta

## ADDED Requirements

### Requirement: Delay, event, and forced models
The library SHALL provide Mackey-Glass, Hutchinson's delayed logistic, delayed Kuramoto, a bouncing ball, Newton's
cradle, a circular billiard, integrate-and-fire and Izhikevich neurons, the forced Duffing oscillator, stochastic
resonance, and a Brownian particle near a wall with multiplicative noise as registered factories.

#### Scenario: Izhikevich regular spiking
- **WHEN** the Izhikevich neuron runs with the regular-spiking parameters and a constant input of 10
- **THEN** it fires periodically with each spike reset recorded as an event
