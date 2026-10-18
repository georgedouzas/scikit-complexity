# co-simulation Specification

## Purpose

Lets one model join subsystems that need different mathematical forms, such as agents and differential equations, by
advancing each subsystem with its own solver and exchanging values between them at synchronisation points.

## Requirements

### Requirement: Systems split into partitions by form
Compiling a system whose law kinds do not fit one form SHALL split it into partitions, each the laws targeting the
states of one subsystem whose law kinds fit one form, and SHALL give each partition its own problem.

#### Scenario: Agents and ODE in sibling subsystems
- **WHEN** a system holds a subsystem of agents with rule laws and a subsystem with rate laws
- **THEN** compiling produces two partitions, one agent-based and one ODE

#### Scenario: Law assigned by its target
- **WHEN** an interaction in the parent system returns one rate law on an ODE state and one rule on an agent group
- **THEN** each law joins the partition that owns its target

### Requirement: Laws read other partitions as inputs
A law SHALL be able to use the states of another partition, which enter its partition as inputs that hold their values
from the last synchronisation point.

#### Scenario: Held input
- **WHEN** an ODE law reads an agent aggregate and the agents change it in the middle of a synchronisation interval
- **THEN** the ODE partition uses the value from the start of the interval until the next synchronisation point

### Requirement: Aggregates summarise groups for other partitions
An aggregate SHALL turn an agent group or element group into a scalar value, the count, the sum, the mean, or a custom
reduction of a variable, that other partitions read like a state by its dotted name.

#### Scenario: Count of agents
- **WHEN** an aggregate counts the agents of the group `farmers` and 12 farmers are alive at a synchronisation point
- **THEN** an ODE law that reads the aggregate sees the value 12 until the next synchronisation point

### Requirement: Agent rules read other partitions
An agent rule SHALL read the states of other partitions by dotted name through its model view.

#### Scenario: Agents react to a continuous resource
- **WHEN** a farmer rule reads `land/soil.fertility` from an ODE partition
- **THEN** it receives the value from the last synchronisation point

### Requirement: The scheduler advances partitions to synchronisation points
The co-simulation scheduler SHALL advance every partition with its own solver to each synchronisation point and exchange
inputs there, with a Gauss-Seidel scheme by default, a Jacobi scheme on request, and a synchronisation interval that
divides the frame duration.

#### Scenario: Interval must divide the frame
- **WHEN** a user sets a synchronisation interval of 0.03 with frames every 0.1
- **THEN** a named validation error is raised that names both values

#### Scenario: Gauss-Seidel order
- **WHEN** a co-simulation uses the Gauss-Seidel scheme
- **THEN** each partition reads the inputs already advanced by the partitions before it in system order

### Requirement: Splitting converges to the joint solution
For a system that also fits one form, co-simulating it as partitions SHALL converge to the joint solution as the
synchronisation interval shrinks, with an error that shrinks at least linearly with the interval.

#### Scenario: Split predator and prey
- **WHEN** a Lotka-Volterra system is split into a predator subsystem and a prey subsystem and co-simulated with
  intervals 0.1, 0.05, and 0.025
- **THEN** the error against the joint ODE solution at time 10 shrinks by a factor of at least 1.8 with each halving of
  the interval, as first-order convergence requires

### Requirement: One runtime covers every partition
A co-simulation SHALL produce frames, trajectories, live parameter updates, reproducible random runs, and Rerun logs
with the same dotted names as a single-form simulation.

#### Scenario: Mixed trajectory
- **WHEN** a user asks the trajectory of a farmer and soil co-simulation for `land/soil.fertility` and the agent
  variable `village/farmers.wealth`
- **THEN** it returns the fertility per frame and the wealth per agent alive at each frame
