# Spec Delta

## ADDED Requirements

### Requirement: Pendulum swing-up as an environment
`pendulum_swing_up` SHALL have an environment that observes the cosine and sine of the angle and the angular velocity,
acts on the pivot acceleration within its bounds, and rewards closeness to the upright position, with the
energy-pumping controller of Åström and Furuta (2000) as its reference.

#### Scenario: Reference controller is rewarded
- **WHEN** the energy-pumping controller runs for 20 seconds from the hanging position
- **THEN** the mean reward over the last half of the run is above the mean reward of the pendulum left hanging
