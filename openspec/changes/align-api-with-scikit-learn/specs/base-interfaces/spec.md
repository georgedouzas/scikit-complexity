## ADDED Requirements

### Requirement: Estimators are the parts and calibration
The classes with `prepare` and `run` SHALL be the parts a modeler defines, which are elements, interactions, laws, and
systems, plus the calibration of parameters to data. Simulating, analysing, and running experiments SHALL be functions
of a system or of its results.

#### Scenario: Listing estimators
- **WHEN** a test lists the public classes of `skcomplex.core` that have `prepare` and `run`
- **THEN** every one is a part or the calibration estimator
