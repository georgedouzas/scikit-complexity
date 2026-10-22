# Spec Delta

## ADDED Requirements

### Requirement: Fluxes are drawn as bars, graphs, and images
A constraint network SHALL by default stream its fluxes each frame as a bar chart in process order, and its objective
values as time series. A network of at most 100 processes SHALL also stream as a graph of pools and processes whose edge
widths scale with the absolute flux. A network in a group on the sites of a lattice SHALL stream a chosen flux per site as an image.

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

### Requirement: Runs indexed by mode or iteration stream on their own timeline
An enumeration or estimation run SHALL log each frame on the `step` timeline and on a sequence timeline named by its
axis, `mode` or `iteration`, and SHALL NOT log `sim_time`. Its blueprint's time panel SHALL use that timeline.

#### Scenario: Modes on a timeline
- **WHEN** the elementary modes of the toy network stream with a Rerun observer
- **THEN** the recording has a `mode` timeline with the values 0 and 1 and no `sim_time` timeline, and the flux bar
  chart and graph show one mode at each value

### Requirement: Labelling fits are drawn as distributions and residuals
A labelling experiment SHALL by default stream, at each iteration, the estimated fluxes on the pool-process graph, the
residual sum of squares as a series, and, per measured fragment, the measured and simulated mass isotopomer
distributions as two bar charts.

#### Scenario: Fit converges in the viewer
- **WHEN** the TCA example estimation streams with a Rerun observer
- **THEN** the entity `<experiment>/rss` holds a series over the `iteration` timeline, and at the last frame the bar
  charts under `<experiment>/mdv/<fragment>/simulated` and `<experiment>/mdv/<fragment>/measured` agree within the
  measurement noise
