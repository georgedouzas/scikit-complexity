# analysis-adapters Specification

## Purpose
Connects the analyses to the data that core model types produce: it names the data kinds, extracts each kind from a
trajectory or an experiment, lists the analyses that apply, and gives every data kind one default analysis.

## Requirements

### Requirement: Analyses are keyed by data kind
Every analysis SHALL declare the one data kind it consumes, series, jump path, field, agent snapshot, network, or
ensemble, SHALL take plain arrays or graphs of that kind, and SHALL NOT read a model, a domain, or a library model name.

#### Scenario: Same function on two domains
- **WHEN** a user passes the last frame of an Ising lattice and the last frame of a Gray-Scott field to the same
  structure factor analysis
- **THEN** both calls succeed with the same signature and return the same result type

#### Scenario: Arrays without a simulation
- **WHEN** a user passes a NumPy array of measured data, with no trajectory, to any analysis of its data kind
- **THEN** the analysis returns its result, and no analysis requires a trajectory or a system

### Requirement: Adapters extract every data kind from a trajectory
The analysis layer SHALL extract from a trajectory its series of named states, reporters, or observables, the jump
paths of discrete states with their jump times, the frames of a lattice or a field in grid shape, the agent snapshots
of a group with positions, velocities, and states, and the network of every frame, each as plain arrays or graphs.

#### Scenario: Series of an ODE
- **WHEN** a user extracts the series of `prey.n` and `predators.n` from a Lotka-Volterra trajectory
- **THEN** the result holds the times and a two-column array equal to `trajectory['prey.n']` and
  `trajectory['predators.n']`

#### Scenario: Jump path of a Gillespie run
- **WHEN** a user extracts the jump path of a birth-death trajectory simulated with Gillespie's algorithm
- **THEN** the result holds the jump times and the state after each jump, with no repeated state between jumps

#### Scenario: Frames of a lattice
- **WHEN** a user extracts the frames of `spins.spin` from a 64 × 64 Ising trajectory of 100 frames
- **THEN** the result has shape `(100, 64, 64)` and equals `trajectory.lattice('spins.spin')`

#### Scenario: Unwrapped agent positions
- **WHEN** a user extracts the positions of random walkers in a periodic continuous space of side 10
- **THEN** the adapter returns positions unwrapped across the boundary, so no step between frames exceeds half the side

#### Scenario: Missing data kind
- **WHEN** a user extracts agent snapshots from an ODE trajectory
- **THEN** a named error says that the trajectory has no agent group and lists the data kinds it has

### Requirement: Adapters extract ensembles from experiments
The analysis layer SHALL extract from an experiment result one array of a named series per run, aligned on common
times, with the design row of every run, so ensemble analyses read replicates and sweeps as one array.

#### Scenario: Replicates of a stochastic run
- **WHEN** a user extracts the ensemble of `x` from an experiment of 200 replicates of an Ornstein-Uhlenbeck process
- **THEN** the result has one row per replicate and one column per common time, and the rows carry their seeds

### Requirement: Applicable analyses are discovered
The analysis layer SHALL list the analyses that apply to a trajectory, a system, an experiment result, or an array,
each with its data kind, the adapter that feeds it, and how it reads the object.

#### Scenario: Lattice model
- **WHEN** a user lists the analyses for an Ising system
- **THEN** the list holds structure factor, correlation length, cluster size distribution, and Binder cumulant under the
  field kind, and series analyses for its observables, and no agent analysis

#### Scenario: Agent model
- **WHEN** a user lists the analyses for a Vicsek trajectory
- **THEN** the list holds polarization, nearest-neighbour statistics, and mean squared displacement under the agent
  kind

#### Scenario: Plain array
- **WHEN** a user lists the analyses for a two-dimensional NumPy array
- **THEN** the list holds the field analyses and the series analyses that read columns, and says which reading each
  assumes

### Requirement: Every data kind has a signature analysis
The analysis layer SHALL compute one default analysis per data kind from a trajectory or an experiment result, and
return its curves with axis labels, a title, and the measured values worth stating, so a gallery example can plot it
without choosing an analysis.

#### Scenario: Signatures per data kind
- **WHEN** a user computes the signature analysis of a trajectory of each data kind
- **THEN** a series gives its power spectrum with the autocorrelation time, a jump path its waiting-time distribution
  with the Fano factor, a field the structure factor of its last frame with the correlation length or the peak
  wavelength, agents the order parameter over time and the nearest-neighbour distance distribution, a network its degree
  distribution, and an ensemble the mean with a 90 percent band over runs

#### Scenario: Model with several data kinds
- **WHEN** a user computes the signature analysis of an agent model on a lattice that also records reporters
- **THEN** the result holds one signature per data kind, in the order agents, field, series

#### Scenario: Gallery figure
- **WHEN** a gallery example of the model library plots its analysis figure without naming an analysis
- **THEN** the figure shows the signature analysis of its own run, with the labelled axes and the title the signature
  carries

### Requirement: Every analysis works in the default install
Every analysis SHALL work in the default install, those backed by a copyleft library such as `deeptime` included,
because every library the core uses is a runtime dependency.

#### Scenario: Default install
- **WHEN** a user lists the analyses of a jump path in the default install
- **THEN** the Markov state model is listed under the path kind and estimates a model from the path
