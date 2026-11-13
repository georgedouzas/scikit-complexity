## MODIFIED Requirements

### Requirement: Custom systems are built from the core

Examples in the README and the user guide SHALL build custom systems from `skcomplex.core`, and SHALL import ready-made
systems as classes from their subfield package `skcomplex.systems.<field>.<subfield>`. They MAY import an analysis of
one field and its result from the family package `skcomplex.systems.<field>.common.<family>` that holds it, and SHALL
import nothing else from field packages.

#### Scenario: Import from a field module

- **WHEN** a Python block of the README or the user guide imports anything but a system class from a package of
  `skcomplex.systems.<field>`, or anything but an analysis or its result from a family package of a field
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A field analysis in a guide page

- **WHEN** the guide page on quantum systems imports `operator_measures` from
  `skcomplex.systems.physics.common.quantum` and names that package in its prose
- **THEN** the documentation test passes

#### Scenario: A field module in the prose

- **WHEN** the prose of the README or the user guide names a package of `skcomplex.systems.<field>` other than a
  subfield package or a family package of the field
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A custom system from the core alone

- **WHEN** a reader runs the first custom system of the README in a fresh interpreter
- **THEN** it simulates, and no module of `skcomplex.systems.<field>` is loaded
