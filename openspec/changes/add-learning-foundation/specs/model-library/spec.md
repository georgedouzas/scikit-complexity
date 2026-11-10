# Spec Delta

## ADDED Requirements

### Requirement: Universal Lotka-Volterra predator and prey
The library SHALL provide `universal_lotka_volterra` in `biology/ecology`, the predator and prey model in a mechanistic
form and in a hybrid form whose predation terms are one learned network, identified from the hare and lynx pelt
records of 1900 to 1920 as distributed with Carpenter (2018), with a gallery example and an inventory entry.

#### Scenario: Mechanistic fit agrees with the published estimates
- **WHEN** the mechanistic form is identified on the pelt records with Gaussian noise on the logarithms
- **THEN** the four rates are within 10% of 0.55, 0.028, 0.80, and 0.024, the posterior means of Carpenter (2018)

#### Scenario: Hybrid form on held-out years
- **WHEN** both forms are identified on the records of 1900 to 1915 and run to 1920
- **THEN** the hybrid form is reported with its training and validation errors, and its validation error on 1916 to
  1920 is at most that of the mechanistic form times 1.5

#### Scenario: Untrained network
- **WHEN** the hybrid form runs with the last layer of its network at zero
- **THEN** the hares grow exponentially at the rate `r` and the lynx decay at the rate `d`, within 1e-10

### Requirement: Outbreak with a learned contact rate
The library SHALL provide `learned_contact_sir` in `biology/epidemiology`, an SIR outbreak in a closed population whose
contact rate is constant or a learned positive function of time, identified from the boys confined to bed in the 1978
influenza outbreak of an English boarding school, observed partially with negative binomial noise, with a gallery
example and an inventory entry.

#### Scenario: Constant form agrees with the published estimates
- **WHEN** the constant form is identified on the confined counts with the negative binomial likelihood
- **THEN** `beta` and `gamma` lie within the 90% posterior intervals of Grinsztajn et al. (2021) for this outbreak

#### Scenario: Learned contact rate
- **WHEN** the learned form is identified on the same counts
- **THEN** its negative log-likelihood is at most that of the constant form, and its contact rate stays positive at
  every time

### Requirement: Cart and pole
The library SHALL provide `cart_pole` in `engineering/control`, the cart and pole of Barto, Sutton, and Anderson (1983)
with the constants of Gymnasium's `CartPole-v1`, an environment as a class attribute, a policy part, a gallery example
that trains the policy, and an inventory entry.

#### Scenario: Pole falls without control
- **WHEN** the system runs from an angle of 0.05 radians with the force held at zero
- **THEN** the terminal observable becomes nonzero before 2 seconds

#### Scenario: Equations of Florian
- **WHEN** one Euler step of 0.02 is taken from a given state and force
- **THEN** the new state equals the step of the equations of Florian (2007) with the constants of the system within
  1e-12
