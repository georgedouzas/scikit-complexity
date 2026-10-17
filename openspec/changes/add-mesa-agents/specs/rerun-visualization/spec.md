# Spec Delta

## ADDED Requirements

### Requirement: Agents are drawn in their space
The Rerun observer SHALL draw agents on a grid or in a continuous space as one batch of points per agent group, and
agents on a network as graph nodes and edges, with colors from an agent variable.

#### Scenario: Schelling grid
- **WHEN** a Schelling segregation model on a 20 by 20 grid is logged
- **THEN** one `Points2D` batch per agent group is logged per frame, at the agents' cell coordinates

#### Scenario: Voter model network
- **WHEN** a voter model on a network is logged
- **THEN** the network is logged as graph nodes and edges, and node colors follow the agents' opinions

### Requirement: Changing populations are logged consistently
The Rerun observer SHALL log agents with their ids, so an agent keeps its identity across frames and removed agents
disappear.

#### Scenario: Agent removed
- **WHEN** an agent is removed between two frames
- **THEN** it is absent from the batch logged at the second frame
