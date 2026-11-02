## ADDED Requirements

### Requirement: Laws read reductions of other partitions
A law SHALL read a reduction of a group of another partition, such as `count(s.farmers)` or
`total(s.farmers.wealth)`, and the co-simulation SHALL compute it at every synchronisation point and hold it fixed
until the next one.

#### Scenario: Count of agents
- **WHEN** an ODE law reads `count(s.farmers)` of an agent group and 12 farmers are alive at a synchronisation point
- **THEN** the law sees the value 12 until the next synchronisation point

## REMOVED Requirements

### Requirement: Aggregates summarise groups for other partitions
**Reason**: Aggregates were reductions declared as parts of a system.
**Migration**: Read the reduction in the law of the other partition.
