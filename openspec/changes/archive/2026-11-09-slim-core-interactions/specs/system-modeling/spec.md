## ADDED Requirements

### Requirement: Core interactions are distinct operations
Every class of `skcomplex.core.components.interactions` SHALL name a mathematical operation that no other class of the
package already gives with one fixed function or argument. A specific form of an operation SHALL be written as the core
class with its function, or as a named realization in the field that has a term for it.

#### Scenario: No parametrised copy
- **WHEN** the classes of `skcomplex.core.components.interactions` are listed
- **THEN** none is a subclass that only fixes the rate, the flux, the guard, or the map function of another core class

#### Scenario: Specific form without a realization
- **WHEN** a mathematics system needs an Ornstein-Uhlenbeck control or a graph diffusion
- **THEN** it writes `RateInteraction` with the rate `(goal - x) / time` or `D laplacian(u)`, with no class of its own
