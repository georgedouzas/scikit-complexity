## ADDED Requirements

### Requirement: Quantities are declared apart from laws
Observables, auxiliaries, invariants, objectives, and aggregates SHALL derive from `BaseQuantity`, and none SHALL be
a law or an element. A component SHALL return its laws from `laws` and its auxiliaries, invariants, and objectives
from `quantities`, and a law SHALL be rejected from `quantities` and a quantity from `laws`, each with a named
error.

#### Scenario: Quantities of a component
- **WHEN** a component returns a rate law from `laws` and an auxiliary and an invariant from `quantities`
- **THEN** the system compiles, the auxiliary is substituted into the rate, and the invariant is checked at every
  frame

#### Scenario: Quantity among the laws
- **WHEN** a component returns an invariant from `laws`
- **THEN** validation raises a named error that names the component and says to declare it in `quantities`

#### Scenario: No quantity is a law
- **WHEN** a test checks `ObservableQuantity`, `AuxiliaryQuantity`, `InvariantQuantity`, `ObjectiveQuantity`, and `AggregateQuantity` against `BaseLaw`,
  `BaseComponent`, and `BaseQuantity`
- **THEN** each is a subclass of `BaseQuantity` only

### Requirement: Noise processes are not elements
`WienerProcess` SHALL derive from no component base, and an interaction SHALL bind it through a role that takes noise
processes.

#### Scenario: Correlated noise
- **WHEN** an interaction binds a `WienerProcess` with covariance `C` and two noise laws draw its processes
- **THEN** the increments over a step `h` have the covariance `h C`, and `WienerProcess` is not a subclass of
  `BaseComponent`
