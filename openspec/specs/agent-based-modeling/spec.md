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
activate agents in random order by default, with a seed that makes runs reproducible.

#### Scenario: Reproducible run
- **WHEN** the same agent-based system is simulated twice with random state 0
- **THEN** the two trajectories are equal

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
