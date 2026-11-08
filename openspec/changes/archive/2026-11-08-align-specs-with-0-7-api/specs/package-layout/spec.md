## MODIFIED Requirements

### Requirement: The library calls its ready-made items systems
Every identifier, path, documentation page, and sentence of the library SHALL call a ready-made item of the catalog a
system, never a model, including scientific names such as "the SIR system". Identifiers of other libraries, such as
`mesa.Model`, and the titles of cited references SHALL keep their own wording.

#### Scenario: Catalog identifiers
- **WHEN** a user lists the public names of `skcomplex.systems`
- **THEN** they are `all_systems` and `import_extra`, and every ready-made system class ends in `System`

#### Scenario: Prose of the documentation
- **WHEN** a reviewer searches the README, the user guide, and the docstrings for the word model as a name of a
  ready-made system
- **THEN** it appears only in identifiers of other libraries and in the titles of cited references
