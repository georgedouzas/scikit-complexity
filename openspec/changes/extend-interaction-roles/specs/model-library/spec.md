# Spec Delta

## ADDED Requirements

### Requirement: Boolean networks take any number of inputs
`random_boolean_network` and `yeast_cell_cycle` SHALL bind the inputs of every node through one variadic role, so a node
reads any number of inputs up to the number of nodes and no input role is bound to the target as padding.

#### Scenario: Eight inputs per node
- **WHEN** `random_boolean_network(n_nodes=12, k=8)` is built and simulated for 50 steps
- **THEN** every node's update reads 8 distinct nodes through a table of 256 entries, and the run completes

#### Scenario: Too many inputs
- **WHEN** `random_boolean_network(n_nodes=4, k=5)` is called
- **THEN** a named validation error is raised that names `k` and the number of nodes

#### Scenario: Yeast cycle unchanged
- **WHEN** `yeast_cell_cycle()` runs from its default start for 15 steps
- **THEN** its trajectory equals the one recorded from the five-role version before the rebuild, and it ends in the
  stationary G1 state with only Cdh1 and Sic1 on

### Requirement: Ring roads are one group of cars
`optimal_velocity` and `intelligent_driver_model` SHALL hold their cars as one group and one interaction that reads the
car ahead by a periodic member offset, for any number of cars.

#### Scenario: One group whatever the number of cars
- **WHEN** `optimal_velocity(n_cars=50)` is built
- **THEN** the system holds one group of 50 cars and one interaction, and its trajectory has the arrays `cars.x` and
  `cars.v` with 50 columns

#### Scenario: Same dynamics as one interaction per car
- **WHEN** `optimal_velocity(n_cars=20)` and `intelligent_driver_model(n_cars=20)` run with their default settings
- **THEN** positions and speeds match those of the per-car versions recorded before the rebuild within a relative
  tolerance of 1e-8

#### Scenario: Jam on the rebuilt ring
- **WHEN** the rebuilt optimal velocity model runs above its critical density from near-uniform flow
- **THEN** the spread of the car speeds grows and a stop-and-go wave forms

### Requirement: Agent models read groups through roles and the space through the model
`trafficking` and `meadow_capacity` SHALL bind every group their rules read to a role, `trafficking` and
`ring_road_drivers` SHALL read the space's extent from the model view, and no rule in the library SHALL read a group by
a hard-coded path or its own parameters by its label.

#### Scenario: Trafficking with two roles
- **WHEN** `trafficking(random_state=0)` is built and run for 90 days
- **THEN** its daily interaction binds the roles `women` and `premises`, has no `width` or `height` argument, and the
  bills and trips equal those recorded before the rebuild

#### Scenario: Nested meadow
- **WHEN** the meadow model built with random state 0 is placed as the subsystem `farm` of another system and run
- **THEN** it runs and its cow counts equal those of the model run at the root

#### Scenario: Ring road without a length parameter
- **WHEN** `ring_road_drivers(10, length=500.0)` runs
- **THEN** the drivers have no `road_length` parameter and every position stays in `[0, 500)`

#### Scenario: No rule reads its own label
- **WHEN** the library's layout test scans the rule functions of every registered model
- **THEN** none reads `model.parameters` with its own label or names a group by a literal path
