# Spec Delta

## Purpose

Lets a user translate system dynamics models from Vensim and XMILE files into model modules built on the core's
interactions and law utilities, which compile, run, and analyse like hand-written models.

## ADDED Requirements

### Requirement: Model files are translated into model modules
`SystemDynamicsTranslator(path)` SHALL, when run, set `source_` to the Python source of a model module that uses only
public names of the library, and `settings_` to the `start`, `until`, `dt`, and `save_every` of the file's control
section. The translator SHALL build the source from the abstract model of PySD's translators, never from the code PySD
generates.

#### Scenario: Teacup runs like PySD
- **WHEN** the teacup model is translated from its Vensim file, saved, imported, and simulated with its settings
- **THEN** the temperature at every saved time is within 1e-6 of PySD's run of the same file with the same time step

#### Scenario: XMILE reads like Vensim
- **WHEN** the same SIR model is translated from a Vensim file and from an XMILE file
- **THEN** the two systems have the same states, parameters, and interactions

### Requirement: Translated modules have one shape
A translated module SHALL define one element class whose states are the stocks and whose parameters are the
constants, a function that returns the system, a function that returns one observable per auxiliary, and the run
settings as module constants.

#### Scenario: Auxiliaries are observables
- **WHEN** the SIR model is translated and simulated with the observables of its module
- **THEN** the trajectory records every auxiliary of the file under its snake-case name

### Requirement: Flows become named transfers
A flow variable that leaves one stock and enters another SHALL become a `TransferInteraction` between the two states
named by the flow, and a flow variable of one stock a `TransferInteraction` with `Source()` or `Sink()`, so the
trajectory reports every flow as `<element>.<flow>`.

#### Scenario: SIR flows are named transfers
- **WHEN** an SIR model whose infection flow leaves susceptible and enters infected is translated and run
- **THEN** the infection is a `TransferInteraction` named after the flow, the trajectory reports it as
  `<element>.infection`, and an invariant on the total population holds

#### Scenario: Loops of a translated model
- **WHEN** a user lists with `LoopAnalysis` the feedback loops of a translated population model with births and deaths
- **THEN** the result holds one reinforcing loop through births and one balancing loop through deaths

### Requirement: Remaining rates become relaxations or rates
The part of a stock's net rate that no flow variable holds SHALL become a `RelaxationInteraction` when it has the
goal-seeking form `(goal - stock) / time`, and a `RateInteraction` otherwise.

#### Scenario: Goal seeking becomes relaxation
- **WHEN** a model whose stock changes by `(target - stock) / adjustment_time` with no flow variable is translated
- **THEN** the stock's rate is a `RelaxationInteraction` with that goal and time, and its run matches PySD within 1e-6

### Requirement: Built-in functions map to the law utilities
Lookups SHALL become `lookup` tables, conditionals `where`, the input signals `step`, `ramp`, `pulse`, and
`pulse_train`, delays and smoothing `delay1`, `delay3`, `smooth`, `distributed_delay`, or `delayed`, `MIN` and `MAX`
`minimum` and `maximum`, and the random functions draws. A start time other than zero SHALL shift `Time` in every
expression.

#### Scenario: Third-order delay
- **WHEN** a model with `DELAY3` of a step input is translated and run
- **THEN** its output is within 1e-6 of PySD's run of the same file

### Requirement: Subscripts become groups
A model whose stocks are all subscripted over one range SHALL become a group with one member per subscript, in the
order of the range, and `SUM` over the range SHALL become a total over the group.

#### Scenario: Regional populations
- **WHEN** a population model subscripted over three regions is translated and run
- **THEN** the trajectory holds one row per region for each stock, equal to PySD's run of the same file within 1e-6

### Requirement: Untranslatable constructs are reported together
Running the translator on a file with constructs that have no counterpart SHALL raise one named error listing every
such construct with its variable: functions such as `GAME`, `ALLOCATE`, and the vector functions, data read from
spreadsheets, macros, variables subscripted over more than one range, and models that mix subscripted and
unsubscripted stocks.

#### Scenario: Unsupported functions
- **WHEN** a file uses `ALLOCATE BY PRIORITY` in one variable and `GET XLS DATA` in another
- **THEN** the error names both variables and both functions
