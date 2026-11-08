## ADDED Requirements

### Requirement: The documentation follows the scikit-learn split
The README, the user guide, every docstring example, and every gallery example SHALL use `simulate`, the analysis
functions, and the experiment functions. A migration page SHALL list every removed name with its replacement.

#### Scenario: A gallery example
- **WHEN** a gallery example runs a system
- **THEN** it calls `simulate` and the analysis functions, and no `Simulation` or `*Analysis` class

### Requirement: Compatibility is documented per kind of part from tags
The documentation SHALL hold one page per kind of part, which are elements, interactions, laws, solvers, backends,
analyses, and experiments. Each page SHALL be generated from the tags, with a table of what each class declares or
accepts and worked, executed examples of a compatible and an incompatible combination with the error it raises. The
single compatibility page SHALL be removed.

#### Scenario: Choosing a solver
- **WHEN** a reader opens the solvers page
- **THEN** it shows, for each solver, the tags it supports, and an example where a mismatched solver raises an error
  naming the tag

### Requirement: The documentation explains the system hierarchy
The documentation SHALL hold a systems page that explains `BaseSystem`, the composite `System`, general systems of the
core, and field realizations, with an executed example of a subclass that changes a ready-made system. The catalog
pages SHALL show, for each system, its class, its import path, its model parameters, and its capability tags.

#### Scenario: A catalog page
- **WHEN** a reader opens the page of `SIRSystem`
- **THEN** it shows `from skcomplex.systems.biology.epidemiology import SIRSystem`, the parameters, and the tags
