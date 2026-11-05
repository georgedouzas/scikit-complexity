# Spec Delta

## Purpose

Lead a reader to one of the two ways to use the library, a prebuilt model or a custom one, and let them find which components
work together before they simulate.

## ADDED Requirements

### Requirement: The documentation offers two paths

The README and the user guide SHALL open by presenting two paths: select a prebuilt model from the catalog, or build a custom
model from the core.

#### Scenario: First screen of the README

- **WHEN** a reader opens the usage section of the README
- **THEN** it shows selecting and running a prebuilt model by name, and building a custom model from the core, each with
  code that runs

### Requirement: Custom models are built from the core

Examples in the README and the user guide SHALL build custom models from `skcomplex.core`, `skcomplex.specs`,
`skcomplex.engine`, and `skcomplex.analysis`, and SHALL reach prebuilt models only through `catalog`, `describe`, `create`, and
`run` of `skcomplex.models`.

#### Scenario: Import from a field module

- **WHEN** a Python block of the README or the user guide imports from `skcomplex.models.<field>`
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A field module in the prose

- **WHEN** the prose of the README or the user guide names `skcomplex.models.<field>`
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A custom model from the core alone

- **WHEN** a reader runs the first custom model of the README in a fresh interpreter
- **THEN** it simulates, and no module of `skcomplex.models.<field>` is loaded

### Requirement: A user can find compatible components

The user guide SHALL explain the order of choices for a system: time and space, elements, interactions, and solver. It SHALL
show how to read what an interaction needs and what an element carries, in code and on the compatibility page.

#### Scenario: Asking before building

- **WHEN** a reader follows the guide to choose an interaction for an element
- **THEN** the guide's code reads the interaction's requirements and the element's states, and shows the two agree

#### Scenario: Getting it wrong

- **WHEN** a reader follows the guide's example that binds an interaction to an element without the state it reads
- **THEN** the guide shows the report that names the role, the state, and the fix
