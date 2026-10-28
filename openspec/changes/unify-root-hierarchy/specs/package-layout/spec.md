## MODIFIED Requirements

### Requirement: Every class derives from the base it is
Every element and group SHALL derive from `BaseElement`, every interaction from `BaseInteraction`, every space from
`BaseSpace`, every law from `BaseLaw`, every problem from `BaseProblem`, every solver from `BaseSolver`, every backend
from `BaseBackend`, every analysis from `BaseAnalysis`, and every spec from `BaseSpec`, and every one of these bases
SHALL derive from `BaseConfig`. Validation SHALL read a domain's kind from the domain.

#### Scenario: Derived classes
- **WHEN** a test checks every class whose name starts with `Base` in `skcomplex.core`, `skcomplex.engine`,
  `skcomplex.analysis`, `skcomplex.experiments`, and `skcomplex.specs` against `BaseConfig`
- **THEN** every check holds

#### Scenario: Formalism domain
- **WHEN** a quantum state declares a `HilbertDomain` and the system is validated
- **THEN** validation reads its kind from the domain

#### Scenario: Solver over a base problem
- **WHEN** a solver is written against `BaseSolver` alone and prepared with a `BaseProblem` of its form
- **THEN** it advances the problem through `run`
