# Spec Delta

## Purpose

Lets a user run lattice laws on triangular lattices and on arbitrary graphs, update sites by graph colorings, and
update the extremal site of a lattice with its neighbours, with the same laws, backends, and visualization as square
lattices.

## ADDED Requirements

### Requirement: Triangular lattices have six ordered neighbours
The modelling layer SHALL define triangular lattices with periodic, fixed, or reflecting edges, and a six-neighbour
neighbourhood ordered counterclockwise, so that the neighbour in direction `k + 3` modulo 6 is opposite to the
neighbour in direction `k`. Every site SHALL have a position in the plane at unit distance from its six neighbours.

#### Scenario: Six neighbours on a torus
- **WHEN** a user asks for the triangular neighbours of the corner site of a periodic 10 × 10 triangular lattice
- **THEN** the result holds six distinct sites, wrapping across both edges, and each lies at distance 1 from the corner
  site in the plane, with the minimum image across the edges

#### Scenario: Opposite directions
- **WHEN** a user steps from any site in direction `k` and then in direction `k + 3` modulo 6
- **THEN** the walk returns to the starting site

#### Scenario: Hexagonal spread
- **WHEN** a rule makes a site burn when it or one of its triangular neighbours burns, from one burning site on a
  periodic 40 × 40 triangular lattice
- **THEN** after `k` steps, for `k` up to 10, exactly the `3k(k + 1) + 1` sites within `k` steps of the start are
  burning

### Requirement: Lattice laws run on graphs
Local rules, drives, relaxation, and Monte Carlo laws SHALL run on lattice groups whose sites are the nodes of a graph,
in the graph's node order. A graph neighbourhood SHALL give the neighbours of a site along its edges, in the graph's
adjacency order, optionally with the site itself first.

#### Scenario: Grid graph matches the square lattice
- **WHEN** the Game of Life runs from the same glider on a periodic 20 × 20 square lattice with the Moore neighbourhood
  and on the graph that joins every site of that lattice to its eight Moore neighbours, for 40 steps
- **THEN** both runs give the same configuration at every step

#### Scenario: Sites of different degree
- **WHEN** a totalistic rule that counts live neighbours runs on a star graph with 5 leaves and a live centre
- **THEN** every leaf counts 1 live neighbour, and the centre counts the live leaves only

### Requirement: Rules and energies see only real neighbours
A function rule on a graph neighbourhood SHALL receive neighbour values padded to the largest degree with the fill
value, and SHALL receive the number of real neighbours of every site when it asks for it. A local energy SHALL sum an
expression of a site and one neighbour over the real neighbours of the site only, on graphs and on lattices alike.

#### Scenario: Neighbour sum ignores padding
- **WHEN** the XY energy `-J Σ cos(θ - θ_j)` is written as a neighbour sum on a star graph with 4 leaves
- **THEN** the energy change of every proposed rotation equals the change summed over the real neighbours, to 1e-12

#### Scenario: Ising on a graph
- **WHEN** the Ising model runs on the periodic 64 × 64 grid graph with a neighbour-sum energy at temperatures from
  1.5 to 3.5 with ensembles
- **THEN** the mean absolute magnetisation drops below one half between temperatures 2.2 and 2.4, as on the square
  lattice

### Requirement: Colored schedules update independent sites together
Local rules and samplers SHALL support a colored schedule, which partitions the sites by a proper coloring of the
graph that joins every site to its neighbourhood, and updates the color classes in turn. The checkerboard schedule
SHALL remain the two-color case on square lattices.

#### Scenario: Coloring a triangular lattice
- **WHEN** a sampler with the colored schedule runs on a periodic 12 × 12 triangular lattice
- **THEN** no two sites of one color class are neighbours, and every site is visited once per sweep

#### Scenario: Checkerboard rejected where it does not apply
- **WHEN** a user asks for the checkerboard schedule on a triangular lattice or a graph
- **THEN** compiling raises a named error that names the law and suggests the colored schedule

### Requirement: Extremal updates change the extremal site and its neighbours
A system SHALL declare an extremal law that finds the site with the smallest or the largest value of a key state,
breaking ties with the simulation's random stream, and applies a rule to that site and its neighbourhood, a given
number of times per step. Every application SHALL be recorded as an event with its time, site, and extremal value.

#### Scenario: Only the neighbourhood changes
- **WHEN** an extremal law that draws new uniform values runs once on a ring of 100 sites with the radius-1
  neighbourhood
- **THEN** exactly the site with the smallest value and its two neighbours change, and the event names that site and
  its old value

#### Scenario: Reproducible extremal runs
- **WHEN** an extremal model runs twice with the same seed on the NumPy and on the Numba backend
- **THEN** all four runs give identical lattices and identical events at every step

### Requirement: Non-square lattices stream as sites
Triangular lattices SHALL stream to Rerun as hexagonal cells at their site positions, and lattice groups on graphs
SHALL stream as nodes colored by state with their edges, which are logged once.

#### Scenario: Triangular forest fire in the viewer
- **WHEN** a forest fire on a triangular lattice runs with Rerun visualization
- **THEN** the recording holds one colored cell per site at its position in the plane at every logged step
