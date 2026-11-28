## ADDED Requirements

### Requirement: Branching avalanches are events
`SelfOrganizedBranchingSystem` SHALL record every avalanche as an `avalanche` event, whose profile holds the active
sites of every generation before the last, and SHALL hold in the state `tree.sigma` the dissipation of the last
avalanche that ended, the active sites that reached the last generation.

#### Scenario: Profiles of the avalanches
- **WHEN** `SelfOrganizedBranchingSystem(0.5, generations=10)` runs 1000 steps
- **THEN** it records at least one event, every profile starts with 1, has at most 10 entries and sums to the size
  of its event

#### Scenario: Balance of dissipation
- **WHEN** the system runs 10^5 steps from `p = 0.3`
- **THEN** the mean dissipation of the avalanches of the second half lies within 10 percent of 1

### Requirement: Forest fires are events
`ForestFireSystem` and `HexagonalForestFireSystem` SHALL record every fire as a `fire` event, the connected burning
sites in space and time, whose size is the number of trees that burn.

#### Scenario: Fires of a driven forest
- **WHEN** `ForestFireSystem((32, 32), growth=0.01, lightning=1e-3)` runs 500 steps
- **THEN** it records fire events whose sizes add up to the burning site steps of the finished fires

### Requirement: Forest fire boundary
`ForestFireSystem(boundary=...)` SHALL take `'periodic'` as the default, `'fixed'`, where no fire crosses the edge,
or `'reflecting'`, and SHALL raise a `ValidationError` for any other boundary.

#### Scenario: Fire on a strip
- **WHEN** a fire starts at column 0 of a full forest on a `(5, 20)` strip with no growth and no lightning
- **THEN** with the periodic boundary column 19 burns at the first step, and with the fixed boundary it burns at
  step 19

### Requirement: Traffic jams are events
`NagelSchreckenbergSystem` SHALL record every jam as a `jam` event, the connected stopped cars in space and time.

#### Scenario: A queue that dissolves
- **WHEN** ten stopped cars stand bumper to bumper on a ring of 100 cells with `slowdown=0.0`
- **THEN** the run records one jam of duration 10, size 55 and profile `10, 9, ..., 1`

### Requirement: Sandpile with random neighbours
`SandpileSystem(neighbors='random')` SHALL send every grain of a toppling site to a site drawn uniformly at every
toppling and keep it with the probability `conservation`, while `neighbors='fixed'` SHALL stay the default. A
`conservation` outside `(0, 1)` with random neighbours SHALL raise a `ValidationError`.

#### Scenario: Mean-field avalanches
- **WHEN** `SandpileSystem((32, 32), neighbors='random', conservation=0.999)` drops 10^5 grains after its transient
- **THEN** no site holds 4 grains at the end of a step, and the fitted size exponent below the cutoff lies within 0.15
  of 3/2

#### Scenario: Conservative random sandpile refused
- **WHEN** `SandpileSystem(neighbors='random', conservation=1.0)` is prepared
- **THEN** a `ValidationError` says that a pile with no loss never stabilises

### Requirement: Bak-Sneppen with random neighbours
`BakSneppenSystem(neighbors='random', k=K)` SHALL renew at every update the least fit species and `K - 1` other
species drawn uniformly afresh, while `neighbors='fixed'` SHALL keep the ring or the graph as the default.

#### Scenario: Threshold of the random-neighbour model
- **WHEN** `BakSneppenSystem(1000, neighbors='random', k=3)` runs past its transient
- **THEN** fewer than 5 percent of the fitnesses lie below `1 / 3`, and the fitnesses above it are uniform within 10
  percent per tenth of the interval
