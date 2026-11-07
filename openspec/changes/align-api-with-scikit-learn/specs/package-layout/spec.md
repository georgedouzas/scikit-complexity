## ADDED Requirements

### Requirement: Estimators for what is defined, functions for what is done
The public API SHALL hold estimators for the parts a modeler defines and for calibration, and public functions for
simulating, analysing, and running experiments. Every other helper SHALL be a private method of the class that uses it.
Compilers and problem forms SHALL be private, with one documented hook for a system package that brings its own form.

#### Scenario: Public surface
- **WHEN** a test lists the public names of `skcomplex.core.engine`
- **THEN** it finds solvers, backends, observables, `simulate`, and `Trajectory`, and no compiler or problem class
