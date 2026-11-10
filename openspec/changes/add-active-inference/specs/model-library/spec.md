## ADDED Requirements

### Requirement: Active-inference flock
The library SHALL provide `ActiveInferenceFlockSystem` in `biology/collective_behaviour` (Heins et al. 2024, arXiv
2307.14804): agents at constant speed with unit headings, each sensing the mean distance to its neighbours in `L`
sectors and its rate of change with Gaussian noise, holding beliefs in generalised coordinates of a model that relaxes
every distance to `eta`, and updating beliefs and heading by gradient descent on the Laplace-encoded free energy.

#### Scenario: One sector gives a social force
- **WHEN** the system runs with one sector and no noise
- **THEN** the change of the heading of every agent in a step equals the precision-weighted prediction error of the
  rate of change of the distance times the mean unit vector to its neighbours, within 1e-10

#### Scenario: Distance set point
- **WHEN** two agents with parallel headings start at a distance of 2 with `eta = 1` and no noise, and run 15 seconds
- **THEN** their mean distance over the last 5 seconds lies within 0.2 of 1

#### Scenario: Defaults of the paper
- **WHEN** a user builds the system with its defaults
- **THEN** it has 50 agents, 4 sectors of 60 degrees from -120 to 120 degrees of the heading, `eta = 1`, `alpha = 0.5`,
  and a step of 0.01 seconds

### Requirement: Collective states of the flock
The system SHALL report the polarization and the normalised angular momentum of the group as observables, and its
default and milling configurations SHALL reproduce the ordering of collective states of Figure 2 of the paper.

#### Scenario: Polarized and milling configurations
- **WHEN** the default configuration and the milling configuration, with the velocity noise variance 2, `eta = 0.5`,
  and `alpha = 0.1`, each run 15 seconds for 20 seeds
- **THEN** the mean polarization over the last 10 seconds is larger in the default configuration, and the mean
  angular momentum is larger in the milling configuration

### Requirement: The flock is recorded in the inventory
The inventory SHALL record `active_inference_flock` as implemented with the source 2307.14804 and the primitive
`reduction-derivatives`, and the gallery SHALL hold its example.

#### Scenario: Inventory and gallery
- **WHEN** the inventory check and the strict gallery coverage run
- **THEN** they report no problem for `active_inference_flock`
