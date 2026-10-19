# Spec Delta

## ADDED Requirements

### Requirement: Fluxes are drawn as bars, graphs, and images
A constraint network SHALL by default stream its fluxes each frame as a bar chart in process order, and its objective
values as time series. A network of at most 100 processes SHALL also stream as a graph of pools and processes whose edge
widths scale with the absolute flux. A network on a lattice SHALL stream a chosen flux per site as an image.

#### Scenario: Flux bar chart
- **WHEN** a static network of 10 processes runs with a Rerun observer
- **THEN** the entity `<network>/fluxes` holds a bar chart of 10 values at every frame, and `<network>/objective` holds
  the objective series

#### Scenario: Flux graph
- **WHEN** the E. coli core network streams with the graph view
- **THEN** the viewer shows its pools and processes as graph nodes with a fixed layout, and the edges of processes
  without flux have the minimal width

#### Scenario: Growth per site
- **WHEN** a lattice network streams its growth flux
- **THEN** the viewer shows a lattice-sized image of the growth flux that updates every frame, zero at inactive sites
