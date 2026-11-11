## RENAMED Requirements

- FROM: `### Requirement: The library generates no part classes`
- TO: `### Requirement: The library generates no component classes`

## MODIFIED Requirements

### Requirement: The library generates no component classes

No module of the library SHALL set `__signature__` or build a component with `type()`.

#### Scenario: Source check

- **WHEN** the test suite scans the modules of the library
- **THEN** it finds no `__signature__` assignment and no component built with `type()`, and the Mesa agent and view
  classes that `MesaSolver` builds are not components
