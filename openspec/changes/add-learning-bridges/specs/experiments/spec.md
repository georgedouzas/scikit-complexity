# Spec Delta

## ADDED Requirements

### Requirement: Identification and calibration select their settings by cross-validation
`identify` and `calibrate` SHALL accept a scikit-learn splitter or a number of folds as `cv`, with `groups` for group
splitters, a scorer as `scoring`, and a parameter grid as `param_grid`. They SHALL fit every setting on the training
runs of every fold, score it on the test runs, fit all the runs again with the best setting, and return a frozen result
that reports `cv_results`, `best_params`, and `best_score`.

#### Scenario: Folds over runs
- **WHEN** `identify` runs with `cv=KFold(5)` on 20 observed runs of a Lotka-Volterra system with a learned term
- **THEN** `cv_results` holds 5 test scores, and every fold fits on 16 runs and scores on 4

#### Scenario: Groups kept whole
- **WHEN** `calibrate` runs with `cv=GroupKFold(3)` and `groups` naming a condition shared by the runs of each of 6
  patients
- **THEN** no fold has runs of one patient in both its training and its test runs

#### Scenario: Grid of learning rates
- **WHEN** `identify` searches 3 learning rates with 3 folds and `scoring='r2'`
- **THEN** `best_params` holds the learning rate of the highest mean test score, `best_score` equals that score, and
  the `system` of the result is a system that `simulate` runs

#### Scenario: Same seed, same search
- **WHEN** the search runs twice with the same `random_state`
- **THEN** both results hold equal `cv_results` scores and equal estimates

#### Scenario: No selection
- **WHEN** `identify` runs without `cv` and `param_grid`
- **THEN** `cv_results`, `best_params`, and `best_score` are `None`
