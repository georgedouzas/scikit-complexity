# Spec Delta

## MODIFIED Requirements

### Requirement: The core holds every domain-agnostic package
The package SHALL expose exactly four subpackages, which together hold every domain-agnostic package and the models:
- `skcomplex.core` holds what a system is: `base`, `space`, `laws`, `components`, `interactions`, and `system`.
- `skcomplex.engine` holds what works on a system: `compile`, `solvers`, `backends`, `runtime`, `viz`,
  `experiments`, and `analysis`.
- `skcomplex.formalisms` holds the formalisms.
- `skcomplex.models` holds the domain models.

Within `core` and within `engine`, each package SHALL import only the packages listed before it. `core` SHALL import
neither `engine` nor `formalisms`, and `engine` SHALL import `core` and not `formalisms`.

#### Scenario: Framework imports
- **WHEN** a user imports `Element` from `skcomplex.core.components`, `Simulation` from `skcomplex.engine.runtime`, `RK4`
  from `skcomplex.engine.solvers`, and `Domain` from `skcomplex.core.base`
- **THEN** every import succeeds

#### Scenario: Old framework paths are gone
- **WHEN** a user imports `skcomplex.model`, `skcomplex.core.model`, `skcomplex.core.compile`, `skcomplex.core.solvers`,
  `skcomplex.core.formalisms`, `skcomplex.solvers`, or `skcomplex.runtime`
- **THEN** the import fails

#### Scenario: Upward import
- **WHEN** a module of `skcomplex.engine.solvers` imports from `skcomplex.engine.backends`
- **THEN** the import contracts of the checks session fail and name the importing module and the imported one

#### Scenario: Core reaching the engine
- **WHEN** a module of `skcomplex.core.components` imports from `skcomplex.engine.compile`
- **THEN** the import contracts of the checks session fail and name the module

### Requirement: Models use only the public core
Modules of `skcomplex.models` SHALL import only public names of the packages of `skcomplex.core`,
`skcomplex.engine`, and `skcomplex.formalisms`, and third-party libraries, never private modules or names.

#### Scenario: Private core import
- **WHEN** a model module imports a private module, such as `skcomplex.engine.compile._compile`
- **THEN** the import contracts of the checks session fail and name the module and the import
