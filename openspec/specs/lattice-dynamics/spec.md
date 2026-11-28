# lattice-dynamics Specification

## Purpose

Lets a user build lattice systems, cellular automata, systems with slow drive and fast relaxation, and Monte Carlo spin
systems, on lattices with ordered neighbourhoods, and run them vectorised.

## Requirements

### Requirement: Lattices define sites and ordered neighbourhoods
The modelling layer SHALL define lattices of any dimension with periodic, fixed, or reflecting edges, and neighbourhoods
as ordered offsets. One ball neighbourhood SHALL give the offsets within a radius under a named metric, Chebyshev,
Manhattan, or hexagonal, with the site itself left out, put first, or kept in its lexicographic place. The Chebyshev and
Manhattan offsets SHALL come in lexicographic order, and the hexagonal offsets ring by ring, counterclockwise. A lattice
space SHALL name the metric of the neighbourhood agents query, and SHALL take no second name for it.

#### Scenario: Moore neighbourhood on a torus
- **WHEN** a user asks for the Chebyshev ball of radius 1 around the corner cell of a periodic 10 × 10 lattice
- **THEN** the result holds the eight cells around it, wrapping across both edges

#### Scenario: Von Neumann neighbourhood
- **WHEN** a user asks for the Manhattan ball of radius 1 in two dimensions
- **THEN** the offsets are `(-1, 0)`, `(0, -1)`, `(0, 1)`, and `(1, 0)`, in that order

#### Scenario: Wolfram order
- **WHEN** a user asks for the Chebyshev ball of radius 1 in one dimension with the site kept in its place
- **THEN** the offsets are `(-1,)`, `(0,)`, and `(1,)`, the order of Wolfram's rule numbers

#### Scenario: Hexagonal ball
- **WHEN** a user asks for the hexagonal ball of radius 1
- **THEN** the offsets are the six triangular neighbours counterclockwise from the direction of the second axis, so
  the offset `k + 3` modulo 6 is opposite to the offset `k`

#### Scenario: Former neighbourhoods are gone
- **WHEN** a user imports `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, `TriangularNeighborhood`,
  or `GeneratorNeighborhood` from `skcomplex.core.components.spaces`
- **THEN** the import fails with `ImportError`, since the project ships no alias for a removed class

### Requirement: Local rules update cells from neighbourhood patterns
A local rule SHALL compute a cell's next value from the ordered values of its neighbourhood, given as a function, a
lookup table, a Wolfram rule number, or a totalistic rule.

#### Scenario: Rule 30
- **WHEN** a user runs Wolfram rule 30 from a single live cell for 15 steps
- **THEN** the space-time diagram matches the published pattern of rule 30

#### Scenario: Glider
- **WHEN** a user runs the Game of Life with a glider on a periodic 20 × 20 lattice for 4 steps
- **THEN** the glider has the same shape, moved one cell diagonally

### Requirement: Update schedules are chosen per rule
Local rules SHALL update synchronously by default, and SHALL support random sequential, random site, and checkerboard
schedules, with stochastic rules drawing from the simulation's random stream.

#### Scenario: Synchronous by default
- **WHEN** a rule copies each cell's left neighbour
- **THEN** one synchronous step shifts the whole configuration by one cell

#### Scenario: Reproducible stochastic rule
- **WHEN** a forest fire runs twice with the same seed
- **THEN** both runs give identical lattices at every step

### Requirement: Relaxation runs to stability within a step
A system SHALL declare drive laws applied once per step and relaxation laws repeated until a stability condition holds,
and SHALL record every relaxation as an event with its size, duration, and area.

#### Scenario: Sandpile avalanches
- **WHEN** the Bak-Tang-Wiesenfeld sandpile on a 50 × 50 lattice runs for 100,000 grains after its transient
- **THEN** no cell exceeds the threshold at the end of any step, and the avalanche size distribution follows a power
  law with an exponent between 1.0 and 1.3 over two decades

### Requirement: Monte Carlo dynamics sample energies
A system SHALL declare a local energy over neighbourhoods and proposed moves, and SHALL sample them with Metropolis,
Glauber, or heat-bath acceptance at a temperature that changes live, with one sweep per unit of time.

#### Scenario: Ising transition
- **WHEN** a user runs the 2D Ising system on a 64 × 64 lattice at temperatures from 1.5 to 3.5 with ensembles
- **THEN** the mean absolute magnetisation drops below one half between temperatures 2.2 and 2.4, around the exact
  critical value 2.269

### Requirement: Lattice systems run fast and stream as images
Lattice rules SHALL run vectorised on the NumPy and Numba backends, and lattices SHALL stream to Rerun as images colored
by state, unless the lattice group or its element type declares another visual.

#### Scenario: Large Game of Life
- **WHEN** a 1000 × 1000 Game of Life runs on the Numba backend with real-time pacing
- **THEN** it sustains 30 steps per second, and the viewer shows the lattice as an image

#### Scenario: Visual of the element type
- **WHEN** an element type declares the visual `VisualSpec('heights')` and a lattice group of it is created without a
  visual argument
- **THEN** the group's visual is `heights`, not an image

#### Scenario: Element type without a visual
- **WHEN** an element type declares no visual and a lattice group of it is created without a visual argument
- **THEN** the group's visual is an image

### Requirement: Lattice steps apply boundary laws
Lattice and Monte Carlo steppers SHALL apply the boundary laws of the problem, including the wrapping of angles, after
every step.

#### Scenario: XY angles stay wrapped
- **WHEN** the XY system on a 16 × 16 lattice runs until 50 with Metropolis rotations
- **THEN** every recorded angle lies in `(-pi, pi]`

### Requirement: Relaxations update their active neighbourhoods
A relaxation that declares its active sites SHALL update, in every iteration after the first, only the active sites
and the sites that read them, and SHALL record the same avalanches as a full update.

#### Scenario: Same avalanches as a full update
- **WHEN** a Bak-Tang-Wiesenfeld sandpile on a 32 × 32 lattice drops 5000 grains from a fixed seed, once with an
  activity and once without
- **THEN** both runs end with the same heights and record avalanches at the same times with the same durations

#### Scenario: Fewer sites evaluated
- **WHEN** the same two runs count the sites their toppling rule evaluates
- **THEN** the run with an activity evaluates fewer than a tenth of the sites of the run without

### Requirement: Monte Carlo copies labels between neighbouring sites
`SamplingInteraction` SHALL sample a state of integer labels on a `LatticeSpace` or a `NetworkSpace` with a region
score and the move `'copy'`, which copies the label of a random neighbour into a random site, `hits` times per site and
sweep, accepted by the Metropolis or the Glauber rule. The region score SHALL be the weighted contact energy between
different labels of neighbouring sites plus elastic constraints `a_l (V_l - A_l)^2` and `b_l (S_l - B_l)^2` on the
volume and the surface of every region, whose strengths and targets are functions of the labels. The sampler SHALL keep
the volume and the surface of every label and update the two changed labels on every accepted copy. A network SHALL
weigh contacts by the edge attribute `'weight'` and site volumes by the node attribute `'volume'`. A region score SHALL
refuse moves other than `'copy'` and the heat-bath rule, and the move `'copy'` SHALL refuse other objectives.

#### Scenario: Volume constraint holds on average
- **WHEN** one region of 9 sites in a medium on a periodic 12 × 12 lattice is sampled with unit contacts and a volume
  constraint of target 30
- **THEN** its volume averages within 2 of 30 after 50 sweeps

#### Scenario: Zero temperature never raises the objective
- **WHEN** four regions are sampled at infinite intensity
- **THEN** the objective recomputed from the labels of every frame never grows

#### Scenario: Weighted networks
- **WHEN** a periodic grid graph with node volumes 0.5 and edge weights 2 holds one region with a volume target of 30
- **THEN** the weighted volume of the region averages within 2 of 30

#### Scenario: Refused moves
- **WHEN** a region score is given the move `'flip'` or the heat-bath rule, or a row score the move `'copy'`
- **THEN** preparing the interaction raises a validation error

### Requirement: Groups on sites read values in the shape of their space
A group on the sites of a lattice or of a network SHALL be a `GroupElement` on `'sites'`, with no second element class
for it. It SHALL take a value in the shape of the lattice, or a mapping from node to value on a network, as one row per
site, and SHALL take the default visual of its space when its members declare none.

#### Scenario: Lattice-shaped values
- **WHEN** a user builds a group on the sites of a 3 × 4 lattice with a state given as a 3 × 4 array
- **THEN** the prepared group has 12 members and the state has one row per site, in row-major order

#### Scenario: Node mapping on a network
- **WHEN** a user builds a group on the sites of a network with a state given as a mapping from node to value
- **THEN** the state has one row per node, in the order of the nodes of the graph

### Requirement: Relaxation records hold their profile
Every cascade record of a relaxation SHALL hold its profile, the number of active members at every round of the
relaxation in order, so that the profile has `duration` entries and sums to `size`.

#### Scenario: Profile of a sandpile avalanche
- **WHEN** a `SandpileSystem((20, 20))` runs 5000 grains
- **THEN** every avalanche event has a profile of `duration` positive integers whose sum is its `size`

#### Scenario: Single toppling
- **WHEN** a relaxation of the Bak-Tang-Wiesenfeld rule starts from a lattice whose only nonempty site holds 4 grains
- **THEN** its record has the size 1, the duration 1 and the profile `(1,)`

### Requirement: Update laws record events of activity
An `UpdateLaw` SHALL take an `activity`, an expression of the values of its member read at the end of every step and
at the start of the run. The run SHALL record every connected set of active member steps as one event under the label
of the law, with its start time, size, duration, area and profile, once no member of it is active.

#### Scenario: One fire through a full forest
- **WHEN** a `ForestFireSystem((10, 10), boundary='fixed', growth=0.0, lightning=0.0)` starts from trees on every site
  and one burning site in a corner
- **THEN** it records one `fire` event of size 100, duration 19, area 100 and profile `1, 2, ..., 10, 9, ..., 1`

#### Scenario: Two fires that never touch are two events
- **WHEN** a forest burns from two sites on opposite sides of an empty column
- **THEN** the run records two `fire` events whose sizes add up to the number of trees

#### Scenario: Unfinished events are left out
- **WHEN** a run ends while a site is still active
- **THEN** the event of that site is not among the events of the trajectory

### Requirement: Active member steps connect by the nearest neighbours of the space
Two active member steps SHALL belong to one event when they are at the same step and nearest neighbours of the space,
or at consecutive steps and the same member or nearest neighbours. A member that is no lattice or network site SHALL
connect only to itself at consecutive steps.

#### Scenario: Jam that moves backwards
- **WHEN** a stopped car at cell `i` is followed at the next step by a stopped car at cell `i - 1`
- **THEN** both belong to one `jam` event

#### Scenario: Runs of one member
- **WHEN** the activity of a member that is no lattice site is `0, 2, 3, 0, 1, 0`
- **THEN** the run records two events, of sizes 5 and 1 and durations 2 and 1
