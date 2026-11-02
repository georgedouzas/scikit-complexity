## MODIFIED Requirements

### Requirement: Laws describe how state changes
An interaction or element SHALL describe its effect as a list of laws of the kinds rate, noise, jump, and update, each
law targeting state variables. Every law kind whose value the engine evaluates numerically SHALL accept a symbolic
expression or a Python callable; equations, quantum generators and dissipators, and Godunov fluxes SHALL accept symbolic
expressions only and say so when given a callable.

#### Scenario: Symbolic rate law
- **WHEN** an interaction returns a rate law on `prey.n` with the expression `-a * predator.n * prey.n`
- **THEN** the compiled system includes that term in the time derivative of the prey count

#### Scenario: Callable rate law
- **WHEN** an interaction returns a rate law whose value is a Python callable of time and current values
- **THEN** the simulation evaluates the callable at each solver evaluation and adds its result to the derivative

#### Scenario: Callable jump change
- **WHEN** a jump law's change is a callable of the current values
- **THEN** each firing applies the change the callable returns

#### Scenario: Callable equation refused
- **WHEN** an equation law receives a callable
- **THEN** the check raises an error naming equations as symbolic only and pointing to a callable rate law
