## ADDED Requirements

### Requirement: Runs take oracles and keep their queries
`System.simulate` SHALL take `oracles` as an argument of the run, and a trajectory SHALL hold `queries`, the records
of every query of the run in order, empty for a run without queries. The `replay` of `simulate` SHALL take a trajectory
recorded with draws, a trajectory with queries, or a query log from `load_queries`, and SHALL force the draws and the
answers it holds.

#### Scenario: Trajectory without queries
- **WHEN** `DeffuantSystem().prepare().simulate(until=10)` runs
- **THEN** its trajectory holds an empty tuple of queries

#### Scenario: Draws and queries replayed together
- **WHEN** a run with named draws and queries is recorded with `simulate(..., record_draws=True)` and replayed with
  `simulate(..., replay=trajectory)` of the same prepared system
- **THEN** the replayed states equal the recorded states and no oracle is called
