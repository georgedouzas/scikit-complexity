# Spec Delta

## ADDED Requirements

### Requirement: Ready-made models are registered factories
Every ready-made model SHALL be a factory function registered in a catalog with its field, description, tags,
references, and default run settings, and the catalog SHALL list, describe, create, and run models by name.

#### Scenario: Browse the catalog
- **WHEN** a user lists the catalog filtered by the field `ecology`
- **THEN** the result holds every ecology model with its description, its references, and the formalism inferred from
  its laws

#### Scenario: Create by name
- **WHEN** a user creates `lotka_volterra` with a predation rate of 0.2
- **THEN** the result is a system that the simulation runs like the same model built from its components

### Requirement: Every registered model runs through the generic interface
Every registered model SHALL be created with its defaults, compiled, simulated, logged to Rerun, and cloned through the
same public interface, without code specific to the model.

#### Scenario: Conformance
- **WHEN** the conformance suite runs over the catalog
- **THEN** every model completes all five steps, and its trajectory holds every declared state and observable

### Requirement: The library uses only the public interface
Library modules SHALL build their models only from names that the core packages export.

#### Scenario: Private import
- **WHEN** a library module imports a private name of another package
- **THEN** the library import check fails and names the module and the import
