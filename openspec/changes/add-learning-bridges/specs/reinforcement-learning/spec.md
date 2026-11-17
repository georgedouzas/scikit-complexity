# Spec Delta

## ADDED Requirements

### Requirement: Policy training selects its settings by cross-validation over conditions
`train_policy` SHALL accept `conditions`, mappings of parameters or initial states that its episodes start from, and,
as `identify` does, `cv`, `groups`, `scoring`, and `param_grid`. It SHALL split the conditions, train every setting on
the training conditions of every fold, score the returns of evaluation episodes from the test conditions, train again
on all conditions with the best setting, and report `cv_results`, `best_params`, and `best_score`.

#### Scenario: Pole lengths held out
- **WHEN** `train_policy` with `'reinforce'` runs on `CartPoleSystem` with 6 pole lengths as conditions, `cv=3`, and 2
  learning rates in `param_grid`
- **THEN** `cv_results` holds 3 test scores for each learning rate, and every fold scores on the 2 pole lengths it did
  not train on

#### Scenario: Selection without conditions
- **WHEN** `train_policy` is called with `cv=3` and no `conditions`
- **THEN** it raises `ValidationError` that says a selection splits conditions and names the argument `conditions`
