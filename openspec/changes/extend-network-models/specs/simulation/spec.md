## ADDED Requirements

### Requirement: Agent simulations take initial states
A simulation of agents SHALL take initial values of the states of an agent group by dotted name, one value per agent
or one shared value, including the state that places an agent on a node or a cell. A name that is no state of the
group SHALL raise a `ValidationError` that names it, and a value of the wrong length SHALL raise one that names the
group size.

#### Scenario: Opinions of voters
- **WHEN** a `VoterSystem` on `nx.star_graph(10)` is simulated with `initial_state={'voters.opinion': x0}`
- **THEN** the opinions of the first frame equal `x0`

#### Scenario: Node of walkers
- **WHEN** a `NetworkRandomWalkSystem` of 5 walkers is simulated with `initial_state={'walkers.node': [0, 0, 1, 1, 2]}`
- **THEN** the walkers sit at those nodes in the first frame

#### Scenario: Wrong length
- **WHEN** `initial_state={'voters.opinion': [1, 0]}` is given to a group of 11 voters
- **THEN** a `ValidationError` names `voters.opinion` and the size 11
