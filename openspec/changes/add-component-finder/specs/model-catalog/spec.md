## ADDED Requirements

### Requirement: Parts are discovered with all_components
`all_components(type_filter=None, package=None)` SHALL return every public element, interaction, law, space,
neighbourhood, and time class of the core and of the systems packages. `type_filter` SHALL select one kind or several,
and `package` SHALL select one package and its subpackages, written as a path below `skcomplex` such as `'core'` or
`'systems/physics'`. `None` SHALL keep every kind or every package.

#### Scenario: Finding a particle
- **WHEN** a user calls `all_components('element', package='systems/physics')`
- **THEN** the result holds `MassPointElement` with its class

#### Scenario: The parts of the core
- **WHEN** a user calls `all_components('interaction', package='core')`
- **THEN** the result holds `RateInteraction` and no class of the systems packages

#### Scenario: An unknown kind or package
- **WHEN** a user calls `all_components('particle')` or `all_components(package='systems/alchemy')`
- **THEN** a `ValidationError` names the value and lists the accepted kinds, or names the package that does not exist

### Requirement: The parts finder returns import paths
Each result of `all_components` SHALL be the import path and the class of a part. The import path SHALL be the dotted
path of the public package the user imports the class from, followed by its name. A class that several packages export
SHALL appear once, under the public package nearest to the module that defines it. The results SHALL be sorted by
import path.

#### Scenario: The line the user writes
- **WHEN** a user finds `MassPointElement`
- **THEN** its import path is `skcomplex.systems.physics.common.particles.MassPointElement`

#### Scenario: Classes with the same name
- **WHEN** two fields each define a public class named `GrowthInteraction`
- **THEN** both appear, each under the import path of its own package

### Requirement: The parts finder imports only what it returns
`all_components` SHALL read the package exports statically and SHALL import only the packages of the classes it
returns.

#### Scenario: Selecting without imports
- **WHEN** a user calls `all_components('element', package='systems/chemistry')` in a fresh interpreter
- **THEN** no module of another field is imported
