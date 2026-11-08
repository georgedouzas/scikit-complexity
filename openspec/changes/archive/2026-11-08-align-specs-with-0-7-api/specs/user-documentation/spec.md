## MODIFIED Requirements

### Requirement: Custom systems are built from the core

Examples in the README and the user guide SHALL build custom systems from `skcomplex.core`, and SHALL import ready-made
systems as classes from their subfield package `skcomplex.systems.<field>.<subfield>`, and nothing else from field
packages.

#### Scenario: Import from a field module

- **WHEN** a Python block of the README or the user guide imports anything but a system class from a package of
  `skcomplex.systems.<field>`
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A field module in the prose

- **WHEN** the prose of the README or the user guide names `skcomplex.systems.<field>`
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A custom system from the core alone

- **WHEN** a reader runs the first custom system of the README in a fresh interpreter
- **THEN** it simulates, and no module of `skcomplex.systems.<field>` is loaded
