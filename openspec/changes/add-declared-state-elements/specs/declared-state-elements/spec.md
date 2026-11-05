# Spec Delta

## Purpose

Let a model declare the states and parameters of an element from data, so that no model generates a class to do it.

## ADDED Requirements

### Requirement: An element declares its variables from data

An element SHALL accept its states and parameters as data at construction, each with its spec, and the compiler SHALL treat
them as it treats annotated constructor arguments.

#### Scenario: Same results as annotated states

- **WHEN** a system is built once with annotated states and once with the same states given as data
- **THEN** both simulations give the same trajectory within 1e-12

#### Scenario: Cloning

- **WHEN** a user clones an element declared from data
- **THEN** the clone has the same states and specs, and a change to the clone's parameters leaves the original unchanged

### Requirement: The library generates no part classes

No module of the library SHALL set `__signature__` or build a part with `type()`.

#### Scenario: Source check

- **WHEN** the test suite scans the modules of the library
- **THEN** it finds no `__signature__` assignment and no part built with `type()`; the Mesa agent and view classes that
  `MesaSolver` builds are not parts

### Requirement: Moved models keep their results

`SparseIdentificationAnalysis`, `next_generation_reservoir`, and the metabolism family SHALL keep their results after moving
to the primitive.

#### Scenario: Reference values

- **WHEN** the moved models run on inputs recorded before the move
- **THEN** their results agree with the recorded values within 1e-9 relative tolerance
