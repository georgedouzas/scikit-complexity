# Spec Delta

## ADDED Requirements

### Requirement: Online and offline activity on a network
The library SHALL provide `online_offline_activity_network` in `social/spreading`, the coupled online engagement and
offline participation of 2605.30432. It SHALL run as a continuous-time Markov process of individuals on a network or
fully mixed, and as its mean-field ODE, with a gallery example and an inventory entry.

#### Scenario: Mean field balances
- **WHEN** the mean-field form runs from `U = 0.99`, `E = 0.01`, `P = 0`
- **THEN** `U + E + D` stays 1 and `P + R` never exceeds `D` within 1e-9, and `E` and `P` vanish at the end

#### Scenario: Large fully mixed population
- **WHEN** the stochastic form runs fully mixed with 10000 individuals over 20 replicates
- **THEN** the mean fractions of `E` and `P` stay within 0.02 of the mean-field solution at every frame

#### Scenario: Weak form from several initial states
- **WHEN** weak-form sparse identification of degree 2 runs on five mean-field runs from different initial `E`, with
  Gaussian noise of 5%
- **THEN** it keeps the terms `U E`, `U P`, `E`, and `P` of the true rates of `U`, `E`, and `P`, with coefficients
  within 5% of the rates of the system
