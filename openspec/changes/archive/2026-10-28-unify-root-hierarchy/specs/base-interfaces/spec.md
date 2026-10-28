## MODIFIED Requirements

### Requirement: Configurable bases share three verbs
Every base class of the library SHALL derive from `BaseConfig` in `skcomplex.base`, which defines `check()`,
`prepare(context)`, and `run(input, final=False)` once, each calling a private hook a child overrides, with trivial
defaults, so every configurable object answers all three. No base SHALL derive from `BaseEstimator` directly, and no
configurable class SHALL add a public verb beyond its constructors and the extension hooks a user implements.

#### Scenario: One vocabulary
- **WHEN** a user calls `check()`, `prepare(...)`, and `run(...)` on a space, a solver, a backend, a visual, an
  observer, a design, and an analysis
- **THEN** every call works, every class is a `BaseConfig`, and none of `initialize`, `advance`, `lower`,
  `on_frame`, `compile`, or `forecast` exists

#### Scenario: A new child through hooks
- **WHEN** a user writes a space that implements `_prepare` setting `size_`, `n_dim_`, and `positions_`, and `_run`
  returning neighbours
- **THEN** the space is checked, prepared, and run through the base's verbs, and the runtime uses it

### Requirement: Values carry fields and a check
Every dataclass of the library outside the models SHALL be a frozen spec that derives from `BaseSpec`, itself a
`BaseConfig`, SHALL be named with the suffix `Spec`, and SHALL be importable from `skcomplex.specs`. A spec SHALL hold
only its fields, `check()`, and readers of its own fields. Validation SHALL call `check()` on every law of a system.

#### Scenario: Invalid law
- **WHEN** a component returns a `RateLaw` whose value is a NumPy array, and the system is validated
- **THEN** the law is created without error, and validation raises an error that names the component and the law

#### Scenario: Terms by value
- **WHEN** a law kind is pickled and loaded in another process
- **THEN** it equals the registered kind with `==`

## ADDED Requirements

### Requirement: Specs live in one package
`skcomplex.specs` SHALL be a package with one module per spec, whose `__init__` exports every spec, so that the
declarations a class reads, the records of a run, and the parts a compiler builds are all imported from it.

#### Scenario: Specs from one place
- **WHEN** a user imports `StateSpec`, `SupportSpec`, `ProblemSpec`, and `FrameSpec` from `skcomplex.specs`
- **THEN** every import succeeds and each class is a frozen `BaseSpec`

#### Scenario: No scattered dataclass
- **WHEN** a test lists every dataclass defined in `skcomplex.core`, `skcomplex.engine`, `skcomplex.analysis`, and
  `skcomplex.experiments`
- **THEN** the list is empty

### Requirement: Acting objects are classes, not dataclasses
An object that keeps state between calls or acts SHALL be a class under the base of its package, deriving from
`BaseConfig`, whose constructor only stores its settings and whose state lives in attributes ending in `_`.

#### Scenario: Observer
- **WHEN** a user inspects the class that records observables during a run
- **THEN** it derives from `BaseConfig` and is not a dataclass

### Requirement: Kinds and protocols are types
Every enum of the library outside the models SHALL live in `skcomplex.types` and end in `Kind`, and every protocol
SHALL live in `skcomplex.types` and end in `Protocol`.

#### Scenario: Kinds from types
- **WHEN** a user imports `NoiseKind`, `OrderKind`, `FormKind`, and `RecordProtocol` from `skcomplex.types`
- **THEN** every import succeeds and no other module of the library defines an enum or a protocol
