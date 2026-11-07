# Spec Delta

## MODIFIED Requirements

### Requirement: Exterior operations check the degree
`PassCompiler` SHALL report, with the code `law-degree`, a codifferential of a site state, an exterior derivative of a
state of the highest degree of a complex, and a sampling of a hyperedge group, each with the path of the law and a fix
that names the operation of the right degree. It SHALL report a coface reduction outside a cell complex with the code
`law-space`. A reduction over a whole group, such as a mean, SHALL add no degree to the value it appears in, while the
exterior operations inside its argument SHALL still be checked.

#### Scenario: Several link issues at once
- **WHEN** a system applies `codifferential` to a site state and samples a hyperedge group
- **THEN** the report holds two issues with the code `law-degree`, each with its own path and fix

#### Scenario: Order parameter of another degree
- **WHEN** a rate on links is scaled by the mean over the faces of a function of the exterior derivative of the links
- **THEN** the report holds no issue with the code `law-degree`

#### Scenario: Misapplied operation inside a mean
- **WHEN** a rate on sites is scaled by the mean of the codifferential of a site state
- **THEN** the report holds an issue with the code `law-degree` for that law
