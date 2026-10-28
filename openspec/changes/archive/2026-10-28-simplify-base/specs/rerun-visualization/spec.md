## ADDED Requirements

### Requirement: Flows and observables of stocks and flows are logged
The observer SHALL log the flows of stocks and flows and the observables of the run as series under the path of their
element, so a view of an observed expression is not blank.

#### Scenario: Hilsa fishery
- **WHEN** a Rerun observer records `hilsa_fishery` for 10 frames with the observables of its model
- **THEN** the recording holds the series `hilsa/adult_weight`

## REMOVED Requirements

### Requirement: Auxiliaries and flows of stocks and flows are logged
**Reason**: Auxiliaries are expressions, which a run records through observables.
**Migration**: Pass an observable for every expression to log.
