# Spec Delta

## ADDED Requirements

### Requirement: Linear programs are drawn as bars and graphs
`LinearProgramVisual` SHALL log the variables of a program as a bar chart under `<path>/variables`, its objective
values as scalars under `<path>/objective`, and, with `graph=True`, the default for at most 100 variables, a graph of
constraints and variables whose edges come from the coefficients and whose widths are proportional to the absolute
values, laid out once with a seeded layout. A program in a site group SHALL be drawn with `ImageVisual` of a chosen
variable per site.

#### Scenario: Bar chart
- **WHEN** a dynamic FBA run streams to Rerun
- **THEN** each frame logs a bar chart of the variables in declared order, and the objective series holds the growth
  rate until the glucose runs out

#### Scenario: Graph
- **WHEN** the toy network runs with `graph=True` twice with the same seed
- **THEN** both recordings hold the same node positions, one node per balance and per variable, with edge widths that
  follow the values

#### Scenario: Value per site
- **WHEN** `comets` streams to Rerun
- **THEN** the growth variable of every site is logged as an image beside the images of the metabolites
