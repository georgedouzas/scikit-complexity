## MODIFIED Requirements

### Requirement: Boolean networks take any number of inputs
`RandomBooleanNetworkSystem` SHALL hold the inputs and tables of all nodes as parameters of one group, and
`YeastCellCycleSystem` SHALL bind the inputs of every node through one variadic role, so a node reads any number of
inputs up to the number of nodes and no input role is bound to the target as padding.

#### Scenario: Eight inputs per node
- **WHEN** `RandomBooleanNetworkSystem(n_nodes=12, k=8)` is built and simulated for 50 steps
- **THEN** every node's update reads 8 distinct nodes through a table of 256 entries, and the run completes

#### Scenario: Too many inputs
- **WHEN** `RandomBooleanNetworkSystem(n_nodes=4, k=5)` is called
- **THEN** a named validation error is raised that names `k` and the number of nodes

#### Scenario: Yeast cycle unchanged
- **WHEN** `YeastCellCycleSystem()` runs from its default start for 15 steps
- **THEN** its trajectory equals the one recorded from the five-role version before the rebuild, and it ends in the
  stationary G1 state with only Cdh1 and Sic1 on
