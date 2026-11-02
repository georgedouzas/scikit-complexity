# Spec Delta

## ADDED Requirements

### Requirement: Cloning samples rare trajectories
The analysis SHALL estimate the scaled cumulant generating function of a time-additive observable by cloning a population of trajectories through the simulation's steppers.

#### Scenario: Poisson counter
- **WHEN** cloning estimates the generating function of the counts of a Poisson process with rate 1 at s = 0.5
- **THEN** it is within 2% of `exp(0.5) - 1`

### Requirement: Uncertainty relation of currents
The analysis SHALL report the precision of a current against its entropy production and check the thermodynamic uncertainty relation.

#### Scenario: Biased random walk
- **WHEN** the current of a biased random walk on a ring is measured
- **THEN** its squared relative fluctuation times the entropy production is at least 2 within the sampling error
