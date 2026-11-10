# Spec Delta

## MODIFIED Requirements

### Requirement: Dependencies run from formalisms to engine to core
The packages of `skcomplex.core` SHALL import downward in the order `experiments`, `analysis`, `data`, `engine`,
`components`, and the modules `base`, `types`, `specs`, and `errors` of `skcomplex.core` below them, so `analysis` never
imports `experiments`, `data` imports neither, `engine` imports none of the three, `components` imports none of them,
and those modules import no package. Within `components` and within `engine`, each package SHALL import only the
packages listed before it.

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.core.engine.backends` imports from `skcomplex.core.engine.solvers`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

#### Scenario: Core reaching the engine
- **WHEN** a module of `skcomplex.core.components.elements` imports from `skcomplex.core.engine.compile`
- **THEN** the import contracts of the checks session fail and name the module

#### Scenario: Analysis reaching experiments
- **WHEN** a module of `skcomplex.core.analysis` imports from `skcomplex.core.experiments`
- **THEN** the import contracts of the checks session fail and name the module

#### Scenario: Data reaching experiments
- **WHEN** a module of `skcomplex.core.data` imports from `skcomplex.core.experiments` or `skcomplex.core.analysis`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: The package holds two subpackages
The package SHALL expose exactly two subpackages, `skcomplex.core` and `skcomplex.systems`, and no module beside them:
- `skcomplex.core.base`, `skcomplex.core.types`, `skcomplex.core.specs`, and `skcomplex.core.errors` hold the root
  class `BaseConfig`, the type aliases, the specs, and the errors.
- `skcomplex.core.components` holds what a system is: `base`, `spaces`, `laws`, `elements`, `interactions`, `system`,
  and the types and specs that name them.
- `skcomplex.core.engine` holds what turns a system into a trajectory: `problems`, `compile`, `backends`, `solvers`,
  `runtime`, `translators`, and `viz`.
- `skcomplex.core.analysis` holds what measures a trajectory, an array, or a system.
- `skcomplex.core.data` holds what turns observations and runs into data: reading, simulating, and splitting observed
  runs, and turning them into arrays.
- `skcomplex.core.experiments` holds what runs many systems.
- `skcomplex.systems` holds the catalog of ready-made systems.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.components.elements`, `simulate` from
  `skcomplex.core.engine.runtime`, `RK4Solver` from `skcomplex.core.engine.solvers`, `entropy` from
  `skcomplex.core.analysis`, `read_runs` from `skcomplex.core.data`, and `sweep` from `skcomplex.core.experiments`
- **THEN** every import succeeds

#### Scenario: Old paths
- **WHEN** a user imports `skcomplex.core.engine`, `skcomplex.core.analysis`, `skcomplex.core.errors`, or `skcomplex.systems`
- **THEN** the import fails, since the old paths keep no alias

