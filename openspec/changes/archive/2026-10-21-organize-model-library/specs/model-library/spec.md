# Spec Delta

## MODIFIED Requirements

### Requirement: Ready-made models are registered factories
Every ready-made model SHALL be a factory function registered in a catalog with its field path, description, tags,
references, and default run settings, and the catalog SHALL list, describe, create, and run models by name.

#### Scenario: Browse the catalog
- **WHEN** a user lists the catalog filtered by the field `biology`, or by the field path `biology/ecology`
- **THEN** the result holds every model of that field or subfield with its description, its references, and the
  formalism inferred from its laws

#### Scenario: Create by name
- **WHEN** a user creates `lotka_volterra` with a predation rate of 0.2
- **THEN** the result is a system that the simulation runs like the same model built from its components

### Requirement: The library uses only the public interface
Library modules SHALL build their models only from the public names of `skcomplex.core` and third-party numerical
libraries used inside laws, and no model package SHALL import another.

#### Scenario: Private import
- **WHEN** a library module imports a private name of another package
- **THEN** the library import check fails and names the module and the import

## ADDED Requirements

### Requirement: Models are organized by scientific field
The library SHALL organize models into packages per scientific field, `physics`, `chemistry`, `biology`, `earth`,
`economics`, `social`, `engineering`, `mathematics`, and `computer_science`, each split into subfield packages. Every
model SHALL live in exactly one subfield.

#### Scenario: Import by field
- **WHEN** a user imports `skcomplex.models.physics.fluids`, `skcomplex.models.biology.ecology`, or
  `skcomplex.models.chemistry.pattern_formation`
- **THEN** each package exports the factories and component types of its subfield, such as `burgers`,
  `lotka_volterra`, and `gray_scott`

#### Scenario: Technique packages are gone
- **WHEN** a user imports `skcomplex.models.fluids`, `skcomplex.models.lattices`, `skcomplex.models.fields`, or
  `skcomplex.models.nonlinear`
- **THEN** the import fails, and every model of those packages is found in its scientific subfield

#### Scenario: One home per model
- **WHEN** the catalog is listed in full
- **THEN** every model name appears once, with a field path of the form `<field>/<subfield>`

### Requirement: Models document their pattern
Every registered model SHALL document how it is built from the interface: its states, parameters, and laws as written,
and its equations as the symbolic layer shows them, so a user can copy it as the pattern for a model of their own.

#### Scenario: Equations in the documentation
- **WHEN** a user reads the documentation page of the field `biology`
- **THEN** every ODE, SDE, map, and field model of the field shows its equations as `equations_of` returns them, and
  every agent, lattice, and Monte Carlo model shows its laws as written

### Requirement: Physics model collection
The physics packages SHALL provide models across mechanics, celestial mechanics, oscillators and synchronization,
statistical mechanics, stochastic processes and soft matter, active matter, fluids, waves, and cosmology, as the model
inventory lists them.

#### Scenario: Kepler orbit
- **WHEN** the Kepler two-body model runs for ten periods
- **THEN** its energy and angular momentum stay within solver tolerance, and its period matches Kepler's third law
  within 0.1%

#### Scenario: Active Brownian particles
- **WHEN** active Brownian particles run in two dimensions with speed `v` and rotational diffusion `D_r`
- **THEN** their mean squared displacement at long times grows as `(4 D_t + 2 v^2 / D_r) t` within 5%

### Requirement: Chemistry model collection
The chemistry packages SHALL provide reaction kinetics, oscillating reactions, and pattern formation models, as the
model inventory lists them.

#### Scenario: Oregonator oscillates
- **WHEN** the Oregonator runs with its standard parameters
- **THEN** it settles on a relaxation limit cycle with a stable period

### Requirement: Biology model collection
The biology packages SHALL provide ecology, epidemiology, evolution, neuroscience, systems biology, physiology, and
collective behaviour models, as the model inventory lists them.

#### Scenario: Repressilator oscillates
- **WHEN** the repressilator runs as ODEs and as a jump process with the same rates
- **THEN** both oscillate, and their periods agree within 10%

#### Scenario: Hodgkin-Huxley spikes
- **WHEN** the Hodgkin-Huxley neuron receives a constant current above threshold
- **THEN** it fires periodically, and its spike threshold current lies within 10% of the published value

### Requirement: Earth science model collection
The earth packages SHALL provide climate, oceanography, atmosphere, and seismology models, as the model inventory
lists them.

#### Scenario: Ice-albedo bistability
- **WHEN** the energy-balance model runs from a warm and from a cold initial temperature at the same solar constant
- **THEN** the runs settle in different stable climates

### Requirement: Economics model collection
The economics packages SHALL provide macroeconomics, markets, finance, and econophysics models, as the model
inventory lists them.

#### Scenario: Kinetic wealth exchange
- **WHEN** the random money exchange model runs with conserved money
- **THEN** the total money stays constant and the wealth distribution approaches an exponential

### Requirement: Social science model collection
The social packages SHALL provide opinion dynamics, segregation, spreading, cooperation, and crowd models, as the model
inventory lists them.

#### Scenario: Bounded confidence clusters
- **WHEN** the Deffuant model runs with confidence bound 0.2 from uniform opinions
- **THEN** opinions converge to a few clusters, about `1 / (2 * 0.2)` of them

### Requirement: Engineering model collection
The engineering packages SHALL provide circuits, control, traffic, and power-system models, as the model inventory
lists them.

#### Scenario: Optimal velocity jams
- **WHEN** the optimal velocity model of cars on a ring runs above its critical density from uniform flow
- **THEN** a stop-and-go wave forms and moves against the direction of travel

### Requirement: Mathematics model collection
The mathematics packages SHALL provide dynamical systems, stochastic processes, network processes, and fractal models,
as the model inventory lists them.

#### Scenario: Standard map
- **WHEN** the Chirikov standard map runs above `K = 0.971635` from an orbit near the golden-mean torus
- **THEN** the orbit diffuses in momentum, and below that value it stays bounded

### Requirement: Computer science model collection
The computer science packages SHALL provide automata, evolutionary computation, and neural network models, as the model
inventory lists them.

#### Scenario: Random Boolean network phases
- **WHEN** Kauffman networks with connectivity 1 and 3 run from random states
- **THEN** perturbations die out at connectivity 1 and spread at connectivity 3
