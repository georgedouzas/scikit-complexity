## MODIFIED Requirements

### Requirement: Runs keep their results as attributes
Every verb of a configurable object SHALL return the object itself, and every hook SHALL return `None`. A run SHALL
keep what it produces in an attribute ending in `_`, named once for each family:
- elements, interactions, and systems keep their laws in `laws_`, and no other part has `laws_`,
- laws keep their contribution in `value_`,
- compilers keep the compiled problem in `problem_`,
- lowerings keep what they lowered in `lowered_`,
- backends keep the lowered functions in `functions_`,
- solvers keep the advanced state in `state_`,
- kernels keep their values in `values_`, `None` for a kernel that changes the state in place,
- searches keep the pairs they find in `pairs_`,
- printers keep the code they print in `code_`,
- observers of a compilation keep what they observed in `observed_`,
- the quantum lowering keeps its kernels in `kernels_`, and
- market protocols keep their trades in `trades_`.

Observers of a run, clocks, and visuals SHALL keep their results in the attributes they already document, and
analyses, experiments, and `simulate` SHALL return their results. The overrides of `run` that name their input, such as `BaseClock.run(elapsed=...)`,
SHALL forward to the base verb and return the object.

#### Scenario: A compiler's problem
- **WHEN** a user prepares and runs a compiler on a system
- **THEN** `run()` returns the compiler, and `problem_` is the compiled problem

#### Scenario: A solver's state
- **WHEN** the runtime runs a prepared solver on one step
- **THEN** `run()` returns the solver, and `state_` is the state vector at the end of the step

#### Scenario: A law's contribution
- **WHEN** a user prepares a rate law and runs it at given values
- **THEN** `run()` returns the law, and `value_` is its rate at those values

#### Scenario: A simulation keeps its named input
- **WHEN** a user runs `simulate(system, until=20)`
- **THEN** the call returns the trajectory, which covers the times up to 20

