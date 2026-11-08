## MODIFIED Requirements

### Requirement: The documentation holds a browsable catalog
The documentation SHALL hold an index of every registered system by field and tag, with its description, generated from
`all_systems()`. Each system page SHALL show how to import the system class and list the element and interaction classes its system
uses.

#### Scenario: Index follows the catalog
- **WHEN** a system is registered
- **THEN** the next documentation build lists it in the index under its field and its tags

#### Scenario: System page
- **WHEN** a reader opens the page of `LotkaVolterraSystem`
- **THEN** it shows `from skcomplex.systems.biology.ecology import LotkaVolterraSystem`, and lists the element and interaction classes of its system
