## ADDED Requirements

### Requirement: Components are discovered with all_components
`all_components(type_filter=None, package=None)` SHALL return every public element, interaction, space, and
neighbourhood class of the core and of the systems packages. `type_filter` SHALL select one kind or several among
`'element'`, `'interaction'`, `'space'`, and `'neighborhood'`, and `package` SHALL select one package and its
subpackages, written as a path below `skcomplex` such as `'core'` or `'systems/physics'`. `None` SHALL keep every kind
or every package.

#### Scenario: Finding a particle
- **WHEN** a user calls `all_components('element', package='systems/physics')`
- **THEN** the result holds `MassPointElement` with its class

#### Scenario: The components of the core
- **WHEN** a user calls `all_components('interaction', package='core')`
- **THEN** the result holds `Interaction` and no class of the systems packages

#### Scenario: An unknown kind or package
- **WHEN** a user calls `all_components('law')` or `all_components(package='systems/alchemy')`
- **THEN** a `ValidationError` names the value and lists the accepted kinds, or names the package that does not exist

### Requirement: The components finder lists no laws
`all_components` SHALL NOT list law classes, since laws are values that interactions state and not components.

#### Scenario: Laws are not components
- **WHEN** a user lists `all_components()`
- **THEN** no result is a subclass of `BaseLaw`, such as `RateLaw` or `ProductionLaw`

### Requirement: The components finder returns import paths
Each result of `all_components` SHALL be the import path and the class of a component. The import path SHALL be the
dotted path of the public package the user imports the class from, followed by its name. A class that several packages
export SHALL appear once, under the public package nearest to the module that defines it. The results SHALL be sorted
by import path.

#### Scenario: The line the user writes
- **WHEN** a user finds `MassPointElement`
- **THEN** its import path is `skcomplex.systems.physics.common.particles.MassPointElement`

#### Scenario: Classes with the same name
- **WHEN** two packages each define a public class named `GrowthInteraction`
- **THEN** both appear, each under the import path of its own package

### Requirement: The components finder imports only what it returns
`all_components` SHALL read the package exports statically and SHALL import only the packages of the classes it
returns.

#### Scenario: Selecting without imports
- **WHEN** a user calls `all_components('element', package='systems/chemistry')` in a fresh interpreter
- **THEN** no module of another field is imported
