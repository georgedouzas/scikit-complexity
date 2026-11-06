# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

### Requirement: Using a model imports only its module
The catalog SHALL live in `skcomplex.systems`: registering, listing, describing, and creating systems and their
simulations by name. `create_system`, `create_simulation`, and `describe` SHALL import the module of the system they
name, the modules it imports, and nothing else of the catalog. The engine SHALL hold no catalog and resolve no name of
the catalog. No function of the catalog SHALL run a simulation.

#### Scenario: One model
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
