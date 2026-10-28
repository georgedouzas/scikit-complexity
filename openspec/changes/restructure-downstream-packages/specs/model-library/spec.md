## ADDED Requirements

### Requirement: Field measures with logic of their own live with their models
A measure named by one field that only picks a choice of a general analysis, such as the Binder cumulant or the polar
order, SHALL have no class of its own: models, examples, and tests SHALL use the general analysis. A field measure with
logic of its own, such as the susceptibility peak or the crackling-noise relation, SHALL be a subclass of the closest
general analysis in the model package whose models it measures.

#### Scenario: Binder cumulant without a class
- **WHEN** a user computes the Binder cumulant of magnetisation samples
- **THEN** they run `DistributionAnalysis('moment-ratio')` with orders 4 and 2, and no model package defines a class for
  it

#### Scenario: Crackling noise with the sandpiles
- **WHEN** a user checks the crackling-noise relation of a sandpile's avalanches
- **THEN** they run `CracklingNoiseAnalysis` from `skcomplex.models.physics.criticality`, a subclass of the event
  analysis

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
several packages of one field share SHALL live in `skcomplex.models.<field>.common`, grouped in family packages such as
`particles`, `quantum`, `institutions`, `learning`, or `markets`, with no loose modules beside the families, and only
that field's packages import it. A package of another field that needs a realization SHALL define its own. A field
whose packages share no code SHALL have no `common` package. The realizations the core holds
today SHALL move as the design's table of moved realizations lists.

#### Scenario: Order book from the library
- **WHEN** a user imports the order book from `skcomplex.models.economics.common.markets`
- **THEN** it is a pairing function for the core's matching law, and its priority scenario holds: the earlier of two
  same-price sell orders trades first

#### Scenario: Potential from the library
- **WHEN** a user imports the Lennard-Jones potential from `skcomplex.models.physics.common.particles`
- **THEN** it is a realization of the core's potential primitive, and the existing LAMMPS agreement tests pass with it

#### Scenario: Realization of one package
- **WHEN** only the climate models use `EarlyWarningAnalysis`
- **THEN** it lives in `skcomplex.models.earth.climate`, and the earth field has no `common` package

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
