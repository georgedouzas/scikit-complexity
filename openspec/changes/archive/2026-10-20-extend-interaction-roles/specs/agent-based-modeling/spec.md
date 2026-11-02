# Spec Delta

## ADDED Requirements

### Requirement: Rules read their owner's parameters and roles
A rule SHALL receive a model view scoped to the interaction or agent group that owns it, from which it reads the
owner's parameters by name and the agents of a group by role or by relative path, so the rule works unchanged when its
system is nested in another, and an agent view SHALL report its group's label as well as its path.

#### Scenario: Parameter of a nested interaction
- **WHEN** an interaction with the parameter `threshold` sits in the subsystem `farm` and its rule reads
  `model.parameter('threshold')`
- **THEN** the rule gets the value of `farm/herding.threshold`, also after a live update of that parameter

#### Scenario: Group by role
- **WHEN** an interaction binds its role `pasture` to the group `farm/patches` and its rule reads
  `model.agents('pasture')`
- **THEN** the rule gets the agents of `farm/patches`

#### Scenario: Unknown name
- **WHEN** a rule reads `model.agents('meadow')` and no role, relative group, or absolute group has that name
- **THEN** a named simulation error is raised that lists the owner's roles and the available groups

#### Scenario: Group label
- **WHEN** a rule reads `group_label` of an agent of the group `farm/cows`
- **THEN** it gets `cows`, and `group` still gives `farm/cows`

### Requirement: Agent interactions bind several groups
An interaction whose laws are agent rules SHALL bind roles to any number of agent groups, whatever their sizes.

#### Scenario: Two agent groups
- **WHEN** an interaction binds the roles `women` and `premises` to two agent groups of 42 and 43 agents
- **THEN** the system compiles to the agent-based form and the rule reads both groups through their roles

### Requirement: Rules read the space geometry
The model view SHALL report the extent of the agent space per axis and whether it wraps, and an agent view SHALL return
the displacement and the distance to another agent or a location, the shortest image on a torus and the hop count on a
network.

#### Scenario: Grid dimensions
- **WHEN** a rule of an agent system on a 30 by 20 toroidal grid reads the model's dimensions and torus flag
- **THEN** it gets `(30, 20)` and true

#### Scenario: Displacement across the seam
- **WHEN** an agent at x = 99 asks its displacement to an agent at x = 1, both at y = 0.5, in a toroidal continuous
  space 100 wide
- **THEN** it gets `(2.0, 0.0)` and the distance 2

#### Scenario: Network distance
- **WHEN** an agent on node 0 of a path graph asks its distance to an agent on node 3
- **THEN** it gets 3, and asking its displacement raises a named space error
