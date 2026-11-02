# lattice-dynamics Specification

## Purpose

Lets a user build lattice models, cellular automata, models with slow drive and fast relaxation, and Monte Carlo spin
models, on lattices with ordered neighbourhoods, and run them vectorised.

## Requirements

### Requirement: Lattices define sites and ordered neighbourhoods
The modelling layer SHALL define lattices of any dimension with periodic, fixed, or reflecting edges, and neighbourhoods
as ordered offsets, with Moore, von Neumann, and radius neighbourhoods built in.

#### Scenario: Moore neighbourhood on a torus
- **WHEN** a user asks for the Moore neighbours of the corner cell of a periodic 10 × 10 lattice
- **THEN** the result holds the eight cells around it, wrapping across both edges

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
- **WHEN** a user runs the 2D Ising model on a 64 × 64 lattice at temperatures from 1.5 to 3.5 with ensembles
- **THEN** the mean absolute magnetisation drops below one half between temperatures 2.2 and 2.4, around the exact
  critical value 2.269

### Requirement: Lattice models run fast and stream as images
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
- **WHEN** the XY model on a 16 × 16 lattice runs until 50 with Metropolis rotations
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
