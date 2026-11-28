## ADDED Requirements

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
