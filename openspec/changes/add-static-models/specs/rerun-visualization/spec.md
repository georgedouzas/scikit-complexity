# Spec Delta

## ADDED Requirements

### Requirement: Sweep axes are timelines
Runs with a sweep SHALL log every frame on a timeline named after the swept parameter, holding its value, besides the
`step` timeline. Time series views of a swept run SHALL use that timeline, so a curve such as survival against dose
reads as a time series against the dose.

#### Scenario: Survival curve
- **WHEN** the `lq_cell_survival` example streams to Rerun
- **THEN** the recording has a `dose` timeline, and the survival fraction is a series over it
