# Spec Delta

## ADDED Requirements

### Requirement: Time series views show the whole run
Every time series view of a default blueprint SHALL show the whole run, whichever visual declares it.

#### Scenario: One-dimensional points
- **WHEN** the default blueprint of an ideal gas in one dimension is built
- **THEN** the time series view of the positions shows the time range from the first frame to the last

#### Scenario: Other states of a points visual
- **WHEN** a points visual in two dimensions also logs a scalar state of its members
- **THEN** the time series view of that state shows the whole run, like the views of scalar elements
