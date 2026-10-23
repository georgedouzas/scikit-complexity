# Spec Delta

## ADDED Requirements

### Requirement: Repeated agent economy models
The library SHALL provide `goods_traders`, `shops`, `barter`, `order_book`, `tragedy_of_the_commons`, and
`ridesharing` as registered factories, each with a curated gallery example. Each SHALL be a `System` under discrete
time with a phases schedule, with agents as elements with rule laws in groups, built on the core's matching, transfer,
and invariant laws and on the trading realizations of `skcomplex.models.economics.realizations`, and none SHALL add a
core primitive.

#### Scenario: Barter reaches the contract curve
- **WHEN** `barter` runs Edgeworth exchange between two traders with Cobb-Douglas preferences until no trade improves
  both
- **THEN** the final allocation lies on the contract curve within 2% of the total endowment

#### Scenario: Commons collapse
- **WHEN** `tragedy_of_the_commons` runs with more herders than the meadow carries and no regulating adaptive parameter
- **THEN** the grass falls below a tenth of its capacity and the herd shrinks after it

#### Scenario: Trades conserve goods
- **WHEN** `goods_traders` runs for 50 steps with an invariant on the total of each good
- **THEN** no invariant is violated, since every trade is a transfer between two agents
