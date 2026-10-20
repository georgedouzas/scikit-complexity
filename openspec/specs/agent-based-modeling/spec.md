# agent-based-modeling Specification

## Purpose

Lets a user describe agent-based models, whose agents follow rules written as code on grids, networks, or continuous
spaces and are born and die during a run, and run them or any existing Mesa model with the framework's runtime.

## Requirements

### Requirement: Rules describe agent behaviour
An agent type SHALL describe its behaviour with rule laws, Python functions that update one agent with access to its
own state, its neighbours, and the model, and a system with rule laws SHALL compile to the agent-based form.

#### Scenario: Agent form inferred
- **WHEN** a system contains only agent groups with rule laws
- **THEN** the compiled problem has the agent-based form

#### Scenario: Rules mixed with rates
- **WHEN** a system contains rule laws and rate laws
- **THEN** compiling raises the existing named error that lists the law kinds and the supported combinations

### Requirement: Agents are born and removed during a run
A rule SHALL be able to create agents of any agent type and remove agents, and frames SHALL report the agents alive at
that frame.

#### Scenario: Population grows
- **WHEN** a rule creates one offspring per agent per step for three steps starting from 10 agents
- **THEN** the frames report 10, 20, 40, and 80 agents

### Requirement: Agents live on grids, networks, and continuous spaces
Agent groups SHALL be placed on a grid with Moore or von Neumann neighbourhoods, on a network, or in a continuous
space, and rules SHALL query the neighbours of an agent in its space.

#### Scenario: Grid neighbours
- **WHEN** an agent in the middle of a 10 by 10 Moore grid asks for its neighbouring cells
- **THEN** it receives the 8 surrounding cells

### Requirement: Agent systems run on Mesa
The Mesa backend SHALL build a Mesa model from an agent-based system, advance it one Mesa step per unit of time, and
activate agents in random order by default, with a seed that makes runs reproducible. Each step SHALL advance Mesa's
clock, so the model time that rules read during the step that produces frame `k` is `k`.

#### Scenario: Reproducible run
- **WHEN** the same agent-based system is simulated twice with random state 0
- **THEN** the two trajectories are equal

#### Scenario: Rules read the model time
- **WHEN** a rule stores `model.time` in a state of its agent at every step of a run until time 3
- **THEN** the state is 1, 2, and 3 in the frames at times 1, 2, and 3, not 0

### Requirement: Existing Mesa models run unchanged
`MesaSystem` SHALL wrap a Mesa model class with its constructor parameters, model reporters, and agent reporters, so the
model runs with the framework's simulation, clocks, observers, and trajectories without changes to its code.

#### Scenario: Wrapped Boltzmann wealth model
- **WHEN** a plain Mesa Boltzmann wealth model is wrapped with a Gini reporter and simulated for 100 steps
- **THEN** the trajectory returns 101 Gini values and the total wealth stays constant

#### Scenario: Live parameter on a wrapped model
- **WHEN** a user updates a declared parameter of a running wrapped model
- **THEN** the model attribute takes the new value before the next step

### Requirement: Agent trajectories hold changing populations
A trajectory of an agent-based run SHALL return model reporters as arrays and agent variables per frame as arrays whose
length is the number of agents alive at that frame, with their agent ids.

#### Scenario: Wealth per agent
- **WHEN** a user asks the trajectory of a wealth model for the agent variable `wealth` at frame 5
- **THEN** it returns one value per agent alive at frame 5 with the matching agent ids

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

### Requirement: Agent states do not shadow Mesa attributes
An agent type SHALL raise a named label error when it is defined with a state named after an
attribute that Mesa or the Mesa backend reads from agents, which are `position`, `pos`, `cell`, `unique_id`, and
`model`. The error SHALL name the attribute and the agent type.

#### Scenario: State named position
- **WHEN** a user defines an agent type with a scalar state named `position`
- **THEN** a named label error is raised when the type is defined, which names `position`, and no run fails later
  with a `TypeError`
