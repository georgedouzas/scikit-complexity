## ADDED Requirements

### Requirement: Toppling matrix of a sandpile lattice
`abelian_sandpile(system, kind='toppling')` SHALL return the sparse toppling matrix `Delta` of a `SandpileSystem`
lattice with open edges, the threshold on the diagonal and minus one for every pair of neighbouring sites. A system
that is no Bak-Tang-Wiesenfeld sandpile with fixed neighbours SHALL raise an `AnalysisError` that names the reason.

#### Scenario: Three by three lattice
- **WHEN** the toppling matrix of `SandpileSystem((3, 3))` is computed
- **THEN** it is 9 by 9, its diagonal holds 4, and it holds 24 entries of minus one, two for every pair of neighbours

#### Scenario: Random neighbours have no toppling matrix
- **WHEN** the toppling matrix of `SandpileSystem(neighbors='random')` is asked for
- **THEN** an `AnalysisError` says that random neighbours change at every toppling

### Requirement: Lattice Green function of a sandpile
`abelian_sandpile(system, kind='green')` SHALL return the inverse of the toppling matrix, whose entry `(i, j)` is the
mean number of topplings at `j` after one grain falls on `i`, and SHALL raise an `AnalysisError` that names
`kind='mean-size'` for a lattice too large to hold the dense inverse.

#### Scenario: Inverse of the toppling matrix
- **WHEN** the Green function of `SandpileSystem((8, 8))` is computed
- **THEN** it is symmetric, its entries are positive, and its product with the toppling matrix is the identity within
  `1e-10`

### Requirement: Dhar's mean avalanche size
`abelian_sandpile(system, kind='mean-size')` SHALL return the mean avalanche size `(1 / N) Σ_ij G_ij` from one linear
solve of the toppling matrix, without forming its inverse.

#### Scenario: Simulation agrees with the formula
- **WHEN** a `SandpileSystem((16, 16))` drops 10^5 grains after a transient of 10^4
- **THEN** the number of topplings per grain lies within 5 percent of the mean size

#### Scenario: Growth with the side
- **WHEN** the mean sizes of the lattices of side 32 and 64 are computed
- **THEN** their ratio lies within 10 percent of 4, the growth `L^2` that Dhar derives

### Requirement: Fast-time activity of cascades
`events(records, kind='activity')` SHALL return, from the cascade records of a run, the activity of every round of
every cascade in the order of the records, each cascade preceded by one round of activity 0 for the drive that starts
it, as one series of fast time.

#### Scenario: Two avalanches
- **WHEN** the records hold the profiles `(1, 2)` and `(3,)`
- **THEN** the series is `0, 1, 2, 0, 3`

#### Scenario: Spectrum of sandpile activity
- **WHEN** the power spectrum of the fast-time activity of 10^5 grains on a `SandpileSystem((50, 50))` is computed
- **THEN** the series has one value per round plus one per avalanche, and its power falls with frequency over two
  decades

### Requirement: Shapes of relaxation records
`events(records, kind='excursions')` SHALL return the profiles of cascade records as their shapes, so `crackling_noise`
collapses the mean shapes of avalanches of a relaxation as it does those of an activity series.

#### Scenario: Sandpile shape collapse
- **WHEN** `crackling_noise` reads the avalanche events of 10^5 grains on a `SandpileSystem((64, 64))` with the
  durations 10, 20 and 40
- **THEN** it returns three mean shapes that rise and fall, and a fitted exponent of the mean size against the duration
  between 1.5 and 2.1
