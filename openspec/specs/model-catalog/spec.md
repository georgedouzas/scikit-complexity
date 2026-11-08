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

### Requirement: The documentation holds a browsable catalog
The documentation SHALL hold an index of every registered system by field and tag, with its description, generated from
`all_systems()`. Each system page SHALL show how to import the system class and list the element and interaction classes its system
uses.

#### Scenario: Index follows the catalog
- **WHEN** a system is registered
- **THEN** the next documentation build lists it in the index under its field and its tags

#### Scenario: System page
- **WHEN** a reader opens the page of `lotka_volterra`
- **THEN** it shows `from skcomplex.systems.biology.ecology import LotkaVolterraSystem`, and lists the element and interaction classes of its system

### Requirement: Ready-made systems are imported from their subfield
Every ready-made system SHALL be a public class of its subfield package, imported as
`from skcomplex.systems.<field>.<subfield> import <Name>System`, and SHALL carry its field, topics, references, and
tuned run settings as class attributes. `simulate` SHALL use the tuned run settings for every argument the call leaves
out.

#### Scenario: Direct use
- **WHEN** a user runs `simulate(SIRSystem())` after `from skcomplex.systems.biology.epidemiology import SIRSystem`
- **THEN** the run uses the class's tuned end time, solver hint, and observables

### Requirement: Systems are discovered with all_systems
`all_systems(field=None, topics=(), tags=None)` SHALL return the name and class of every ready-made system, in the role
of `sklearn.utils.all_estimators()`. Selecting by field or topics SHALL read the class definitions statically and import
only the modules of the systems it returns. Filtering by capability tags SHALL import the systems it filters.

#### Scenario: Listing by field without imports
- **WHEN** a user calls `all_systems(field='physics/mechanics')` in a fresh interpreter
- **THEN** the systems of that field are returned, and no module of another subfield is imported

#### Scenario: Filtering by capability
- **WHEN** a user calls `all_systems(tags={'randomness': 'stochastic', 'space': 'network'})`
- **THEN** every listed system has those tags
