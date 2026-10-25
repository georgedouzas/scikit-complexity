# Spec Delta

## Purpose

Lets a user read system dynamics models from Vensim and XMILE files into systems that compile, run, and analyse like
hand-written ones.

## ADDED Requirements

### Requirement: Model files are read into systems
`read_vensim(path)` and `read_xmile(path)` SHALL return the `System` of the model in the file and its run settings,
`start`, `until`, `dt`, and `save_every`, taken from the file's control section. They SHALL build the system from the
abstract model of PySD's translators, never from the code PySD generates.

#### Scenario: Teacup runs like PySD
- **WHEN** the teacup model is read from its Vensim file and simulated with its settings
- **THEN** the temperature at every saved time is within 1e-6 of PySD's run of the same file with the same time step

#### Scenario: XMILE reads like Vensim
- **WHEN** the same SIR model is read from a Vensim file and from an XMILE file
- **THEN** the two systems have the same states, parameters, and laws

### Requirement: Stocks, flows, and auxiliaries keep their meaning
Every stock SHALL become a state, every constant a parameter, and every auxiliary an `Auxiliary` recorded under its
name. A flow that leaves one stock and enters another SHALL become a `TransferLaw` between them, a flow of one stock a
transfer with the environment, and the rest of a stock's net rate a `RateLaw`.

#### Scenario: SIR flows are transfers
- **WHEN** an SIR model whose infection flow leaves susceptible and enters infected is read
- **THEN** the infection is a `TransferLaw` from susceptible to infected, and the conservation check of the compiled
  system holds the total population constant

#### Scenario: Loops of an imported model
- **WHEN** a user lists the feedback loops of an imported population model with births and deaths
- **THEN** the result holds one reinforcing loop through births and one balancing loop through deaths

### Requirement: Built-in functions map to the library's laws
Lookups SHALL become `lookup` tables, `IF THEN ELSE` SHALL become `where`, the input signals SHALL become `step`,
`ramp`, `pulse`, and `pulse_train`, delays and smoothing SHALL become `distributed_delay` of their order or `delayed`
for a fixed delay, and the random functions SHALL become draws. A start time other than zero SHALL shift `Time` in
every expression.

#### Scenario: Third-order delay
- **WHEN** a model with `DELAY3` of a step input is read and run
- **THEN** its output is within 1e-6 of PySD's run of the same file

### Requirement: Subscripts become groups
A variable subscripted over one range SHALL become a group with one member per subscript, in the order of the range,
and `SUM` over the range SHALL become a total over the group.

#### Scenario: Regional populations
- **WHEN** a population model subscripted over three regions is read and run
- **THEN** the trajectory holds one row per region for each stock, equal to PySD's run of the same file within 1e-6

### Requirement: Untranslatable constructs are reported together
Reading a file with constructs that have no counterpart SHALL raise one named error listing every such construct with
its variable: functions such as `GAME`, `ALLOCATE`, and the vector functions, data read from spreadsheets, macros, and
variables subscripted over more than one range.

#### Scenario: Unsupported functions
- **WHEN** a file uses `ALLOCATE BY PRIORITY` in one variable and `GET XLS DATA` in another
- **THEN** the error names both variables and both functions
