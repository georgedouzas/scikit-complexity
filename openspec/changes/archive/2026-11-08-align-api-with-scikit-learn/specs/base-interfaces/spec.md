## ADDED Requirements

### Requirement: Estimators are the parts and calibration
The classes with `prepare` and `run` SHALL be the parts a modeler defines, which are elements, interactions, laws, and
systems, plus the calibration of parameters to data. Simulating, analysing, and running experiments SHALL be functions
of a system or of its results.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `prepare` and `run`
- **THEN** every one is a part or the calibration estimator

### Requirement: Systems form a class hierarchy
`BaseSystem` SHALL be the base of every system: it SHALL hold `prepare`, `run`, `get_tags()` derived from the parts,
nesting, and the passing of model parameters down to the parts. `System(BaseSystem)` SHALL be the generic composite of
elements, interactions, and nested systems. A class SHALL NOT be chosen by mathematical form, which the tags give.

#### Scenario: Generic composite
- **WHEN** a user builds `System(elements=[...], interactions=[...])`
- **THEN** it is a `BaseSystem`, and its tags come from its parts

### Requirement: Ready-made systems are classes with model parameters
Every ready-made system SHALL be a class deriving from `BaseSystem` or from a general system of the core, whose
constructor only stores the model's own parameters and whose `prepare` builds its parts from them. `get_params`,
`set_params`, and `clone` SHALL work on those parameters.

#### Scenario: Changing a model parameter
- **WHEN** a user calls `SIRSystem(beta=0.3).set_params(beta=0.5).prepare()`
- **THEN** the prepared system's infection rate is 0.5, and `get_params()` returns `beta=0.5`

### Requirement: General systems enter the core only when fields share them
A general system of the core SHALL be named after its mathematics, such as `MassActionSystem`, and SHALL enter the
core only when systems of two fields share its structure. A ready-made system SHALL subclass only general systems of
the core or systems of its own field.

#### Scenario: Shared structure
- **WHEN** an epidemic model and a chemical kinetics model both build compartments with mass-action rates
- **THEN** both subclass `MassActionSystem` of the core, and neither imports the other's field
