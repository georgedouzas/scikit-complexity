# Spec Delta

## ADDED Requirements

### Requirement: Observables are logged
The Rerun logger SHALL log every observable as a time series under the entity path of the system that declares it.

#### Scenario: Live order parameter
- **WHEN** a simulation of oscillators with the observable `order` streams to Rerun
- **THEN** the viewer shows `order` as a time series under the system's entity path
