## MODIFIED Requirements

### Requirement: Ready-made systems are classes
Every ready-made system SHALL be a class on `BaseSystem`, or on a general core system such as `ReactionNetworkSystem`, in
the module of its subfield. It SHALL take its model parameters as arguments of its constructor and SHALL give its field
path, topics, references, and tuned run settings as class attributes. `all_systems()` SHALL find every such class.

#### Scenario: Browse the catalog
- **WHEN** a user calls `all_systems(field='biology')` or `all_systems(field='biology/ecology')`
- **THEN** the result holds the name and class of every system of that field or subfield, and each class gives its
  field, topics, and references as class attributes

#### Scenario: Create by name
- **WHEN** a user builds `LotkaVolterraSystem(a=0.2)`, imported from `skcomplex.systems.biology.ecology`
- **THEN** the result is a system that `simulate` runs like the same system built from its components, to the tuned
  end time of its class

### Requirement: Repeated agent economy systems
The library SHALL provide `GoodsTradersSystem`, `shops`, `barter`, `order_book`, `TragedyOfTheCommonsSystem`, and
`ridesharing` as system classes, each with a curated gallery example. Each SHALL be a `CompositeSystem` under discrete
time with a phases schedule, with agents as elements with rule laws in groups, built on the core's matching, transfer,
and invariant laws and on the trading realizations of `skcomplex.systems.economics.realizations`, and none SHALL add a
core primitive.

#### Scenario: Barter reaches the contract curve
- **WHEN** `barter` runs Edgeworth exchange between two traders with Cobb-Douglas preferences until no trade improves
  both
- **THEN** the final allocation lies on the contract curve within 2% of the total endowment

#### Scenario: Commons collapse
- **WHEN** `TragedyOfTheCommonsSystem` runs with more herders than the meadow carries and no regulating adaptive parameter
- **THEN** the grass falls below a tenth of its capacity and the herd shrinks after it

#### Scenario: Trades conserve goods
- **WHEN** `GoodsTradersSystem` runs for 50 steps with an invariant on the total of each good
- **THEN** no invariant is violated, since every trade is a transfer between two agents
