## ADDED Requirements

### Requirement: Field measures are realizations of general analyses
A measure named by one field, such as the Binder cumulant or the polar order, SHALL be a realization in that field's
models: a subclass of a general analysis that only fixes settings and documents the field's name, adding no formula,
verb, or attribute. It SHALL follow the placement rule of realizations: with the one package that measures it, or in
the field's shared realizations when several packages use it.

#### Scenario: Binder cumulant of a model
- **WHEN** a user runs `BinderCumulantAnalysis` from `skcomplex.models.physics.statistical_mechanics` on magnetisation
  samples
- **THEN** it sets the same attributes as the general distribution analysis of kind `moment-ratio` with orders 4 and 2,
  with the same values

#### Scenario: Polar order of a flock
- **WHEN** a user runs `PolarOrderAnalysis` from `skcomplex.models.physics.active_matter` on a Vicsek trajectory
- **THEN** it equals the alignment analysis of kind `mean-vector` on the headings of the flock

### Requirement: Specialized code moves to the model library
When a library package is generalised, its specialized code that still means something in a field, such as a field's
measure, a field's naming, or a composite of general operations, SHALL move to the model library as realizations of the
general classes, and SHALL NOT be removed. The model library SHALL serve as worked examples of how to extend the library
packages by subclassing their bases and general classes.

#### Scenario: Crackling noise kept
- **WHEN** the avalanche analysis dissolves into the general event analysis
- **THEN** the crackling-noise relation stays available as a realization of the criticality models, with the same
  values as before

## MODIFIED Requirements

### Requirement: Specific realizations live in the model library
The library SHALL hold every specific realization of a core primitive or a general analysis in its own private module
of the model package that uses it, next to the model definitions, exported from that package. A realization that
several packages of one field share SHALL live in `skcomplex.models.<field>.realizations`, grouped in family packages
such as `particles`, `quantum`, `institutions`, `learning`, or `markets`, with no loose modules beside the families,
and only that field's packages import it. A package of another field that needs a realization SHALL define its own. A
field whose realizations no two packages share SHALL have no `realizations` package. The realizations the core holds
today SHALL move as the design's table of moved realizations lists.

#### Scenario: Order book from the library
- **WHEN** a user imports the order book from `skcomplex.models.economics.realizations.markets`
- **THEN** it is a pairing function for the core's matching law, and its priority scenario holds: the earlier of two
  same-price sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.models.physics.realizations.particles`
- **THEN** it is a realization of the core's potential primitive, and the existing LAMMPS agreement tests pass with it

#### Scenario: Realization of one package
- **WHEN** only the climate models use `EarlyWarningAnalysis`
- **THEN** it lives in `skcomplex.models.earth.climate`, and the earth field has no `realizations` package

### Requirement: Feedback controllers are engineering realizations
The library SHALL provide PID and state-feedback controllers in `skcomplex.models.engineering.control`, the package of
the control models that use them, as components whose laws read states and set bounded adaptive parameters during a
run, with the integral of the PID error in an element, so their internal states are part of the compiled system and
the core holds no controller.

#### Scenario: Inverted pendulum
- **WHEN** a cart with an inverted pendulum runs from a tilt of 0.1 radians with a tuned PID controller
- **THEN** the angle returns within 0.01 radians of upright within 5 seconds

#### Scenario: No controller in the core
- **WHEN** the field-term and import-graph tests of the core run
- **THEN** they pass, and the controllers are importable only from the control models
