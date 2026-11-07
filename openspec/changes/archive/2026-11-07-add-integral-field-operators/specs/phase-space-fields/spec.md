# Spec Delta

## ADDED Requirements

### Requirement: Simplex domains of class axes
`SimplexSpace` SHALL be a phase space whose class axes are the free frequencies of a simplex, each on `[0, 1]` with the
same number of cells, holding the points whose frequencies sum to at most one. Every cell SHALL carry the fraction of
its volume inside the simplex and every face the fraction of its area inside, so the cells along the slanted face are
partial and the cells beyond it are empty.

#### Scenario: Volumes of the 2-simplex
- **WHEN** a user creates a simplex space with two class axes of 10 cells each
- **THEN** 45 cells are whole, the 10 cells along the slanted face hold half a cell, the other cells are empty, and the
  volumes sum to the area one half

### Requirement: Fluxes and integrals respect partial cells
Finite-volume fluxes on a simplex SHALL pass only through the open part of every face and change a cell by its volume,
so the total is conserved and no density enters the empty cells. Reductions and integrals over the class axes SHALL
weight every cell by its volume. The compile report SHALL hold a local operator other than a finite-volume flux on a
simplex as a `law-space` issue.

#### Scenario: Replicator flow conserves the density on the simplex
- **WHEN** a uniform density on the 2-simplex is advected by the replicator dynamics of three strategies
- **THEN** its total stays constant within 1e-10 relative error and the empty cells stay zero

#### Scenario: Integral of a uniform density
- **WHEN** a density of two on the 2-simplex is reduced over both class axes
- **THEN** the result is one within 1e-12
