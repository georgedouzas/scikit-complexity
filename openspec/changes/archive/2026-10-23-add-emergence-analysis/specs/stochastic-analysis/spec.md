# Spec Delta

## ADDED Requirements

### Requirement: Ergodicity breaking is diagnosed from ensembles
The analysis SHALL diagnose ergodicity breaking from an ensemble of trajectories of an observable: the ergodicity-
breaking parameter, the normalised variance of the time averages across the ensemble; the Thirumalai-Mountain
fluctuation metric as a function of time; and the total-variation distance between the time-averaged distribution of an
agent trait and its ensemble distribution. It SHALL read trajectories through the existing adapters.

#### Scenario: Ergodic diffusion in a box
- **WHEN** the ergodicity-breaking parameter of the positions of 200 Ornstein-Uhlenbeck processes is computed over
  times much longer than their relaxation time
- **THEN** it is below 0.01, and the fluctuation metric decays as `1/t` within a 10% fitted exponent

#### Scenario: Frozen disorder
- **WHEN** the same diagnostics run on 200 trajectories that each stay in one of two wells for the whole run
- **THEN** the ergodicity-breaking parameter stays above 0.5 and the fluctuation metric levels off
