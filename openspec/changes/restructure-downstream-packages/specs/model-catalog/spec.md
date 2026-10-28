## MODIFIED Requirements

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
