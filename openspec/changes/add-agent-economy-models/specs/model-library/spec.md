# Spec Delta

## ADDED Requirements

### Requirement: Repeated agent economy models
The library SHALL provide `goods_traders`, `shops`, `barter`, `order_book`, `tragedy_of_the_commons`, and
`ridesharing` as registered factories built on the existing phase, market, and agent-space primitives, each with a
curated gallery example.

#### Scenario: Barter reaches the contract curve
- **WHEN** `barter` runs Edgeworth exchange between two traders with Cobb-Douglas preferences until no trade improves
  both
- **THEN** the final allocation lies on the contract curve within 2% of the total endowment

#### Scenario: Commons collapse
- **WHEN** `tragedy_of_the_commons` runs with more herders than the meadow carries and no institution
- **THEN** the grass falls below a tenth of its capacity and the herd shrinks after it
