## ADDED Requirements

### Requirement: Stable fits report the S0 location
`fit_law(law='stable')` SHALL report the location in the S0 parametrisation, and SHALL report a skewness of 0 when the
index is 2.

#### Scenario: Index near 1
- **WHEN** samples of stable laws of index 0.98 and 1.02 with S0 location 0 are fitted
- **THEN** both fitted locations are close to 0

#### Scenario: Gaussian data
- **WHEN** normal samples are fitted
- **THEN** the index is close to 2 and the skewness is 0

### Requirement: Branching ratio estimates are quiet
`events(kind='branching')` SHALL print nothing to standard output, and SHALL pass the messages of its estimator on as
Python warnings.

#### Scenario: Captured output
- **WHEN** a branching ratio is estimated while standard output is captured
- **THEN** nothing is captured

### Requirement: Box counting checks grids and counts points
`scaling(kind='box-counting')` SHALL raise an `AnalysisError` for a grid thinner than its largest box, and SHALL count
an array of points at every box size.

#### Scenario: Thin grid
- **WHEN** a grid of shape `(1, 512)` is box counted with boxes larger than 1
- **THEN** an `AnalysisError` names the side that is too short

#### Scenario: Points of a Sierpinski triangle
- **WHEN** 50000 chaos game points of the Sierpinski triangle are box counted as points
- **THEN** the dimension is close to log 3 / log 2

### Requirement: Detrended fluctuation windows are unique
`scaling(kind='detrended-fluctuation')` SHALL use default window sizes without duplicates.

#### Scenario: Default windows
- **WHEN** the fluctuation of a short series is estimated with the default windows
- **THEN** the reported window sizes are strictly increasing
