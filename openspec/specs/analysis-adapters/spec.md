# analysis-adapters Specification

## Purpose
Connects the analyses to the data that core system types produce: it names the data kinds, extracts each kind from a
trajectory or an experiment, lists the analyses that apply, and gives every data kind one default analysis.

## Requirements

### Requirement: Analyses are keyed by data kind
Every analysis SHALL declare the one data kind it consumes, series, jump path, field, agent snapshot, network, or
system, SHALL measure plain arrays or graphs of that kind, and SHALL NOT read a system, a domain, or a
library system name.

#### Scenario: Same function on two domains
- **WHEN** a user runs the same structure factor analysis on the last frame of an Ising lattice and on the last
  frame of a Gray-Scott field
- **THEN** both runs succeed with the same settings and set the same result fields

#### Scenario: Arrays without a simulation
- **WHEN** a user runs any analysis of a data kind on a NumPy array of measured data, with no trajectory
- **THEN** the analysis runs and sets its result fields, and no data analysis requires a trajectory or a system

### Requirement: Adapters extract every data kind from a trajectory
Every analysis SHALL extract from a trajectory, while it runs, its series of named states, reporters, or
observables, the jump paths of discrete states with their jump times, the frames of a lattice or a field in grid
shape, the agent snapshots of a group with positions, velocities, and states, or the network of every frame, as plain
arrays or graphs. The extraction SHALL NOT be a public function.

#### Scenario: Series of an ODE
- **WHEN** a user runs a series analysis with `variables=['prey.n', 'predators.n']` on a Lotka-Volterra trajectory
- **THEN** the analysis reads the times and a two-column array equal to `trajectory.values('prey.n')` and
  `trajectory.values('predators.n')`

#### Scenario: Jump path of a Gillespie run
- **WHEN** a user runs a path analysis on a birth-death trajectory simulated with Gillespie's algorithm
- **THEN** the analysis reads the jump times and the state after each jump, with no repeated state between jumps

#### Scenario: Frames of a lattice
- **WHEN** a user runs a field analysis on `spins.spin` of a 64 × 64 Ising trajectory of 100 frames
- **THEN** the analysis reads frames of shape `(100, 64, 64)` equal to `trajectory.lattice('spins.spin')`

#### Scenario: Unwrapped agent positions
- **WHEN** a user runs an agent analysis on random walkers in a periodic continuous space of side 10
- **THEN** the analysis reads positions unwrapped across the boundary, so no step between frames exceeds half the side

#### Scenario: Missing data kind
- **WHEN** a user runs an agent analysis on an ODE trajectory
- **THEN** a named error says that the trajectory has no agent group and lists the data kinds it has

### Requirement: Applicable analyses are discovered
The analysis layer SHALL list the analysis functions that apply to a trajectory, a system, or an array, each with its data
kind and how it reads the object.

#### Scenario: Lattice system
- **WHEN** a user lists the analyses for an Ising system
- **THEN** the list holds the structure factor, correlation length, cluster size, and Binder cumulant analyses under the
  field kind, and series analyses for its observables, and no agent analysis

#### Scenario: Agent system
- **WHEN** a user lists the analyses for a Vicsek trajectory
- **THEN** the list holds the order parameter, nearest-neighbour, and mean squared displacement analyses under the agent
  kind

#### Scenario: Plain array
- **WHEN** a user lists the analyses for a two-dimensional NumPy array
- **THEN** the list holds the field analyses and the series analyses that read columns, and says which reading each
  assumes

### Requirement: Every data kind has a signature analysis
The analysis layer SHALL hold one signature analysis that, called on a trajectory, runs the default analysis of each
of its data kinds and keeps their curves with axis labels, a title, and the measured values worth stating, so a gallery
example can plot it without choosing an analysis.

#### Scenario: Signatures per data kind
- **WHEN** a user runs the signature analysis on a trajectory of each data kind
- **THEN** a series gives its power spectrum with the autocorrelation time, a jump path its waiting-time distribution
  with the Fano factor, a field the structure factor of its last frame with the correlation length or the peak
  wavelength, agents the order parameter over time and the nearest-neighbour distance distribution, and a network its
  degree distribution

#### Scenario: System with several data kinds
- **WHEN** a user runs the signature analysis on an agent system on a lattice that also records reporters
- **THEN** the analysis holds one signature per data kind, in the order agents, field, series

#### Scenario: Gallery figure
- **WHEN** a gallery example of the system library plots its analysis figure without naming an analysis
- **THEN** the figure shows the signature analysis of its own run, with the labelled axes and the title the signature
  carries

### Requirement: Every analysis works in the default install
Every analysis SHALL work in the default install, those backed by a copyleft library included, because every library
the core uses is a runtime dependency.

#### Scenario: Default install
- **WHEN** a user lists the analyses of a jump path in the default install
- **THEN** the Markov state model is listed under the path kind and estimates a system from the path
