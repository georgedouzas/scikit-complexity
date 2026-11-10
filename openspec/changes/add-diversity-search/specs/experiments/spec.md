## ADDED Requirements

### Requirement: Diversity search is an experiment
The experiments layer SHALL hold `diversity_search(system, parameters, *, descriptors, objective, method, ...)`, which
runs the system at parameter rows inside the bounds and returns a frozen `DiversitySearchResult` with the archive of
parameters, descriptors, and objectives, the coverage, the quality-diversity score, the nearest-neighbour diversity,
and the history over iterations. An unknown method SHALL raise `ExperimentError` naming the methods.

#### Scenario: Unknown method
- **WHEN** a user calls `diversity_search` with `method='random-walk'`
- **THEN** it raises `ExperimentError`, whose message names `'map-elites'`, `'cma-me'`, `'novelty'`, and
  `'goal-exploration'`

#### Scenario: Same seed
- **WHEN** a user runs one search twice with the same random state
- **THEN** both results hold the same archive

### Requirement: Quality diversity by pyribs
The methods `'map-elites'` and `'cma-me'` SHALL fill a grid archive of the descriptor space through pyribs, keeping in
every cell the parameters of the best objective, and `'novelty'` SHALL keep the runs whose descriptors lie far from
their nearest neighbours in a proximity archive.

#### Scenario: Map of a known function
- **WHEN** a user searches a system whose two descriptors equal its two parameters in `[0, 1]`, with a 10 by 10 grid
  and 2000 runs of `'map-elites'`
- **THEN** the coverage exceeds 0.95

#### Scenario: Novelty without an objective
- **WHEN** a user runs `'novelty'` with no objective on the same system
- **THEN** the archive holds runs in every quarter of the descriptor square

### Requirement: Goal exploration samples goals in the descriptor space
The method `'goal-exploration'` SHALL draw a goal in the descriptor bounds, take the archived run whose descriptors lie
nearest to it, perturb its parameters, run them, and archive the result, after a first batch of random rows.

#### Scenario: Goals reached on a known map
- **WHEN** a user explores a system whose descriptors are a smooth invertible map of its parameters
- **THEN** after 500 runs the mean distance from 100 fresh goals to their nearest archived descriptors is below a tenth
  of the width of the bounds
