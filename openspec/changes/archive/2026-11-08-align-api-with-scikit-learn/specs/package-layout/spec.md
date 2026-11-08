## ADDED Requirements

### Requirement: Estimators for what is defined, functions for what is done
The public API SHALL hold estimators for the parts a modeler defines and for calibration, and public functions for
simulating, analysing, and running experiments. Every other helper SHALL be a private method of the class that uses it.
Compilers and problem forms SHALL be private, with one documented hook for a system package that brings its own form.

#### Scenario: Public surface
- **WHEN** a test lists the public names of `skcomplex.core.engine`
- **THEN** it finds solvers, backends, observables, `simulate`, and `Trajectory`, no concrete compiler or `Simulation`,
  and of compilation and problems only the bases a system package's own form builds on, the problem of the shared pass
  core, and the frames observers read

#### Scenario: A system package brings its own form
- **WHEN** a system class tunes the run setting `compiler` with a compiler its package defines on `BaseCompiler`
- **THEN** `simulate` compiles the system with it, while the library's own forms, such as the rewriting and operator
  forms, are chosen from what the system holds
