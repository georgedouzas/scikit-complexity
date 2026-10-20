# spatial-fields Specification

## Purpose

Lets a user write continuous fields over lattices and networks with differential operators and boundary conditions,
simulate them as ODE systems by the method of lines, and couple them with agents.

## Requirements

### Requirement: Fields hold one value per site
The modelling layer SHALL define fields with one value per site of a lattice or node of a network, whose laws are
written for one site.

#### Scenario: Uniform decay
- **WHEN** a field on a 50 × 50 lattice decays with `Rate(u, -k * u)` from a uniform value
- **THEN** every site follows the exponential decay of the single-site equation

### Requirement: Differential operators lower to stencils
Laws SHALL use the Laplacian, gradient, divergence, and advection of fields, which stay symbolic in the equations and
lower to finite differences on lattices and to the graph Laplacian on networks.

#### Scenario: Heat kernel
- **WHEN** a point of heat diffuses on a periodic 1D lattice of 400 sites with spacing 0.05
- **THEN** after time 0.5 the profile matches the Gaussian heat kernel within 1% of its peak

#### Scenario: Diffusion on a network
- **WHEN** a field diffuses on a connected network
- **THEN** the sum of the field stays constant and every node converges to the mean

### Requirement: Fields declare boundary conditions
Fields SHALL take periodic, zero-flux, or fixed-value boundary conditions per state.

#### Scenario: Zero flux conserves mass
- **WHEN** a field diffuses on a lattice with zero-flux edges
- **THEN** the total over the lattice stays constant within 1e-10 relative error

### Requirement: Explicit steps are checked for stability
The compiler SHALL warn when an explicit stepper's step exceeds the diffusive stability limit of a field, and name the
largest stable step and the implicit SciPy methods BDF and Radau as the stiff path. An adaptive explicit method SHALL
warn once when one call spans more than 1000 times the stability limit.

#### Scenario: Too large a step
- **WHEN** a user runs diffusion with RK4 and a step four times the stability limit
- **THEN** a warning names the limit before the run starts

#### Scenario: Stiff path named
- **WHEN** the stability warning is raised for a finite-difference or finite element field
- **THEN** it names BDF and Radau of the SciPy ODE solver

#### Scenario: Adaptive explicit method on a stiff field
- **WHEN** Fisher-KPP with a stability limit near 1e-3 runs with RK45 and frames of 2
- **THEN** one warning names BDF and Radau on the first frame, and later frames do not repeat it

### Requirement: Fields couple with agents
Agents SHALL read a field at their cell and deposit amounts into it, with the field and the agents exchanging values at
synchronisation points.

#### Scenario: Pheromone trail
- **WHEN** foraging ants deposit pheromone that diffuses and evaporates
- **THEN** after the run, the field along the shortest path between nest and food is higher than elsewhere

### Requirement: Fields stream as images
Fields SHALL stream to Rerun as images colored by a color map with a declared range.

#### Scenario: Gray-Scott in the viewer
- **WHEN** a Gray-Scott model streams to Rerun
- **THEN** the viewer shows each field as an image that updates every frame
