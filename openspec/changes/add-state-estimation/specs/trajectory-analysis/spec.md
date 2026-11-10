# Spec Delta

## ADDED Requirements

### Requirement: Ensemble forecasts are scored by proper scores
`forecast_skill` SHALL accept an ensemble forecast, with `metric='crps'` SHALL return the continuous ranked probability
score of the ensemble at every lead time, and for every ensemble SHALL return the coverage of the central intervals of
the levels given and their interval scores.

#### Scenario: Gaussian ensemble
- **WHEN** an ensemble of 10000 draws of a standard normal variable is scored against the true value 0
- **THEN** the score is within 0.005 of 0.2337, the score of a standard normal forecast of its mean

#### Scenario: Calibrated coverage
- **WHEN** 1000 true values and ensembles are drawn from one normal distribution
- **THEN** the coverage of the 90% central interval is within 0.03 of 0.9

#### Scenario: One member
- **WHEN** the ensemble holds one member
- **THEN** the score equals the absolute error of that member within 1e-12
