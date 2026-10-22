# Spec Delta

## Purpose

Lets a user draw quenched random parameters from a simulation's seed, average exactly over enumerated configurations
with Boltzmann weights, and place random points whose contact clusters are observed.

## ADDED Requirements

### Requirement: Quenched parameters are drawn from the seed
Elements SHALL declare quenched parameters by a distribution and a shape. A run SHALL draw them once, before its first
frame, from the simulation's random state, and SHALL keep them fixed during the run. Replicates of an ensemble SHALL
draw independent values.

#### Scenario: Same seed
- **WHEN** a model with 2^10 Gaussian quenched energies runs twice with `random_state=3`
- **THEN** both runs hold identical energies, and a run with `random_state=4` holds different ones

#### Scenario: Disorder average
- **WHEN** an ensemble runs 10 replicates of a model with quenched parameters
- **THEN** each replicate holds its own draw, and the ensemble summary averages observables over the draws

### Requirement: Log-sum-exp reduces over members without overflow
The core SHALL provide the log-sum-exp reduction of an expression over the members of a group, computed without
overflow for values of any size. The library SHALL provide, in `skcomplex.models.physics.realizations`, the logarithm
of the partition function and the Boltzmann average of a value at a temperature, built on it.

#### Scenario: Random energy model entropy
- **WHEN** the entropy per spin `(ln Z + U/T)/N` of 2^20 Gaussian energies of variance `N/2` is averaged over 10
  draws
- **THEN** it is within 0.02 of `ln 2 - 1/(4 T^2)` at `T = 1` and `T = 2`, and below 0.1 at `T = 0.3`, below the
  glass temperature `1/(2 sqrt(ln 2))`

#### Scenario: Large energies
- **WHEN** the energies are of order 10^4 and the temperature is 1
- **THEN** the logarithm of the partition function is finite and equals the largest `-E/T` plus a correction of at
  most `ln` of the number of members

### Requirement: Random point patterns are quenched groups
Point processes SHALL place a group of points in the system's continuous space by a Poisson process of given
intensity, or by a fixed count, drawn from the seed. The points SHALL carry an arrival order, so that a sweep of the
intensity reveals the first points of one pattern and the patterns of a sweep are nested. A point process SHALL
require a continuous space, and validation SHALL report one outside it.

#### Scenario: Nested sweep
- **WHEN** a sweep raises the intensity of a pattern from 0.5 to 1.5
- **THEN** every frame's points are the first points of the next frame's points

#### Scenario: Outside a continuous space
- **WHEN** a point process is placed in a system with a lattice space
- **THEN** validation reports the point process and suggests a `ContinuousSpace`

### Requirement: Contact clusters are observable
The analysis layer SHALL compute the clusters of a point pattern whose members touch within a radius: the number of
clusters, the size of the largest, and whether a cluster spans the space between two opposite faces. It SHALL build
observables of them that a simulation or catalog entry records per frame.

#### Scenario: Spanning probability
- **WHEN** disks of radius 1 are placed in a square of side 60 at filling factors 1.0 and 1.25 over 50 draws each
- **THEN** the fraction of draws with a spanning cluster is below 0.1 at 1.0 and above 0.9 at 1.25
