## MODIFIED Requirements

### Requirement: Translated modules have one shape
A translated module SHALL hold one system class, a function that returns one observable per auxiliary, and the run
settings as module constants. For a file whose stocks are not subscripted and change only through flow variables, the
class SHALL subclass `StockFlowSystem` and give its stocks and flows through its hooks. Any other file SHALL define one
element class whose states are the stocks and whose parameters are the constants.

#### Scenario: Auxiliaries are observables
- **WHEN** the SIR system is translated and simulated with the observables of its module
- **THEN** the trajectory records every auxiliary of the file under its snake-case name

#### Scenario: Flows only become a stock-flow system
- **WHEN** a file whose every stock changes only through flow variables, such as the SIR system, is translated
- **THEN** the module defines a subclass of `StockFlowSystem` and no element class, and its run matches PySD within
  1e-6

#### Scenario: A remaining rate keeps the element shape
- **WHEN** a file whose stock changes by `(target - stock) / adjustment_time` with no flow variable is translated
- **THEN** the module defines an element class of the stocks and a `RateInteraction`, and no subclass of
  `StockFlowSystem`
