# system-catalog Specification

## Purpose
Lists the registered systems and creates, describes, and runs them by name, importing only the systems that are used.

## Requirements

### Requirement: Listing systems imports no system
Importing `skcomplex.systems` and listing the catalog SHALL import no system module and no formalism. Each listed entry
SHALL give the name, field path, description, and tags that the system's registration declares.

#### Scenario: Catalog without imports
- **WHEN** a user imports `skcomplex.systems` and calls `catalog()` in a fresh interpreter
- **THEN** every registered system is listed, and no module of `skcomplex.systems.<field>` and no module of
  `skcomplex.formalisms` is loaded

#### Scenario: Filter by field
- **WHEN** a user calls `catalog(field='physics/mechanics')` in a fresh interpreter
- **THEN** the systems of that field path are listed and no system module is imported

### Requirement: Using a system imports only its module
The catalog SHALL live in `skcomplex.systems`: registering, listing, describing, and creating systems and their
simulations by name. `create_system`, `create_simulation`, and `describe` SHALL import the module of the system they
name, the modules it imports, and nothing else of the catalog. The engine SHALL hold no catalog and resolve no name of
the catalog. No function of the catalog SHALL run a simulation.

#### Scenario: One system
- **WHEN** a user calls `create_system('pendulum')` from `skcomplex.systems` in a fresh interpreter
- **THEN** the system is created, and no module of another subfield and none of `qutip` and `mesa` is loaded

#### Scenario: Unknown name
- **WHEN** a user calls `create_system('no_such_system')` or `create_simulation('no_such_system')`
- **THEN** a named error is raised that names the system and suggests the closest registered names

#### Scenario: No catalog in the engine
- **WHEN** a user lists the public names of `skcomplex.core.engine.runtime`
- **THEN** none of them registers, lists, describes, or creates systems by name

#### Scenario: No run function
- **WHEN** a user lists the public names of `skcomplex.systems`
- **THEN** there is no `run` and no `create`, only `create_system` and `create_simulation` create by name

### Requirement: The listing agrees with the registrations
The listed entry of every system SHALL equal the entry its registration makes when its module is imported. A
registration whose name, field, description, or tags cannot be read without running the module SHALL be rejected.

#### Scenario: Listing matches registration
- **WHEN** every system module is imported after `catalog()` listed them
- **THEN** each registered entry has the name, field path, description, and tags its listed entry had

#### Scenario: Computed registration argument
- **WHEN** a system registers with a field path built at run time, such as a variable
- **THEN** listing the catalog raises a named error that names the system module and the argument

### Requirement: The documentation holds a browsable catalog
The documentation SHALL hold an index of every registered system by field and tag, with its description, generated from
`catalog()`. Each system page SHALL show how to select the system by name and list the element and interaction classes its system
uses.

#### Scenario: Index follows the catalog
- **WHEN** a system is registered
- **THEN** the next documentation build lists it in the index under its field and its tags

#### Scenario: System page
- **WHEN** a reader opens the page of `lotka_volterra`
- **THEN** it shows `create('lotka_volterra')`, and lists the element and interaction classes of its system

### Requirement: A catalog simulation carries the tuned settings
`create_simulation(name, **parameters)` SHALL return an unprepared simulation of the named system that holds the
system as `system` and the registered run settings as its own parameters: the end time `until`, the time step, the
solver, the observables, the initial state, and the seed `random_state=0`. It SHALL run with `prepare` and `run`, and
its settings SHALL change with `set_params`.

#### Scenario: Default run
- **WHEN** a user calls `create_simulation('sir').prepare().run()`
- **THEN** the trajectory ends at the registered `until` of `sir`, with its registered time step and observables

#### Scenario: Changed settings
- **WHEN** a user calls `create_simulation('sir').set_params(dt=0.05).prepare().run(until=20)`
- **THEN** the run uses the step 0.05 and ends at time 20

#### Scenario: The system of a simulation
- **WHEN** a user calls `create_simulation('sir', beta=0.3)`
- **THEN** its `system` is the system `create_system('sir', beta=0.3)` returns

#### Scenario: Reproducible default
- **WHEN** a user runs `create_simulation('poisson_process')` twice
- **THEN** both trajectories are equal, and setting `random_state=None` gives a fresh run

#### Scenario: Unknown run setting
- **WHEN** a registration declares a run setting that is not an argument of `Simulation`
- **THEN** `create_simulation` raises an argument error that names the system and the setting
