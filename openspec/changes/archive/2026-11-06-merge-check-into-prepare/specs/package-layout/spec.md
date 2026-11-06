## MODIFIED Requirements

### Requirement: Every engine class follows the common API
Every class of the engine SHALL derive from exactly one base of its package, SHALL store only its settings in
`__init__`, SHALL validate them in `prepare`, SHALL take the one object it serves in `prepare(target)` and set the
attributes that end in `_`, and SHALL take the one input it acts on in `run(inputs)`. A base SHALL take no argument in
its verbs that its peers do not take, and a variation between subclasses SHALL be a setting.

#### Scenario: Compiling a system
- **WHEN** a user builds the system compiler, checks it, prepares it on a system, and runs it
- **THEN** `prepare` sets `report_` with the issues of the system, and `run` returns the compiled problem or raises the
  report's errors

#### Scenario: A user kernel
- **WHEN** a developer subclasses the kernel base for a new law kind, with the pass that builds it and a solver that
  runs it
- **THEN** the kernel is prepared on the problem and run on the state like every built-in kernel

### Requirement: Engine dataclasses are frozen values
Every dataclass of the engine SHALL be a frozen value with the hook `_check` and plain readers only, and a class that acts SHALL
be a class of the engine's tree instead.

#### Scenario: No acting dataclass
- **WHEN** a reviewer lists the dataclasses of the engine
- **THEN** every one is frozen and has no public verb besides `prepare` and `run`, and no method besides the hook `_check` and plain readers
