## MODIFIED Requirements

### Requirement: An unknown variant lists only the accepted values

When an argument of a component takes one of several named values, or a spec a subclass handles, an unknown value
SHALL raise a `ValidationError` that names the component, the argument, the value given, and the accepted names. The
message SHALL describe an accepted spec in words and SHALL NOT print the name of a base class.

#### Scenario: An unknown mode of a map interaction

- **WHEN** `MapInteraction(..., mode='add')` is prepared
- **THEN** the `ValidationError` names `'add'` and the accepted names `'replace'` and `'increment'`, and its text holds no `BaseSpec`
