## ADDED Requirements

### Requirement: Runs take oracles and keep their queries
`simulate` SHALL take `oracles`, and a trajectory SHALL hold `queries`, the records of every query of the run in
order, empty for a run without queries. `replay` SHALL take a trajectory recorded with draws, a trajectory with
queries, or a query log from `load_queries`, and SHALL force the draws and the answers it holds.

#### Scenario: Trajectory without queries
- **WHEN** the stochastic Deffuant system runs 10 steps
- **THEN** its trajectory holds an empty tuple of queries

#### Scenario: Draws and queries replayed together
- **WHEN** a run with named draws and queries is recorded with `record_draws=True` and replayed from its trajectory
- **THEN** the replayed states equal the recorded states and no oracle is called
