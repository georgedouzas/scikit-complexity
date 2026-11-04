# model-catalog Specification

## Purpose
Lists the registered models and creates, describes, and runs them by name, importing only the models that are used.

## Requirements

### Requirement: Listing models imports no model
Importing `skcomplex.models` and listing the catalog SHALL import no model module and no formalism. Each listed entry
SHALL give the name, field path, description, and tags that the model's registration declares.

#### Scenario: Catalog without imports
- **WHEN** a user imports `skcomplex.models` and calls `catalog()` in a fresh interpreter
- **THEN** every registered model is listed, and no module of `skcomplex.models.<field>` and no module of
  `skcomplex.formalisms` is loaded

#### Scenario: Filter by field
- **WHEN** a user calls `catalog(field='physics/mechanics')` in a fresh interpreter
- **THEN** the models of that field path are listed and no model module is imported

### Requirement: Using a model imports only its module
The model catalog SHALL live in `skcomplex.models`: registering, listing, describing, creating, and running models by
name. `create`, `run`, and `describe` SHALL import the module of the model they name, the modules it imports, and
nothing else of the model library. The engine SHALL hold no catalog and resolve no model name.

#### Scenario: One model
- **WHEN** a user calls `create('pendulum')` from `skcomplex.models` in a fresh interpreter
- **THEN** the system is created, and no model module of another subfield and none of `qutip` and `mesa` is loaded

#### Scenario: Unknown name
- **WHEN** a user calls `create('no_such_model')`
- **THEN** a named error is raised that names the model and suggests the closest registered names

#### Scenario: No catalog in the engine
- **WHEN** a user lists the public names of `skcomplex.engine.runtime`
- **THEN** none of them registers, lists, describes, creates, or runs models by name

### Requirement: The listing agrees with the registrations
The listed entry of every model SHALL equal the entry its registration makes when its module is imported. A
registration whose name, field, description, or tags cannot be read without running the module SHALL be rejected.

#### Scenario: Listing matches registration
- **WHEN** every model module is imported after `catalog()` listed them
- **THEN** each registered entry has the name, field path, description, and tags its listed entry had

#### Scenario: Computed registration argument
- **WHEN** a model registers with a field path built at run time, such as a variable
- **THEN** listing the catalog raises a named error that names the model module and the argument
