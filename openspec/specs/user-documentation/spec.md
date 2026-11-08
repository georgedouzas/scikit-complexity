# user-documentation Specification

## Purpose

Lead a reader to one of the two ways to use the library, a prebuilt system or a custom one, and let them find which components
work together before they simulate.

## Requirements

### Requirement: The documentation offers two paths

The README and the user guide SHALL open by presenting two paths: select a ready-made system from the catalog, or build
a custom system from the core.

#### Scenario: First screen of the README

- **WHEN** a reader opens the usage section of the README
- **THEN** it shows selecting and simulating a ready-made system by name, and building a custom system from the core,
  each with code that runs

### Requirement: Custom systems are built from the core

Examples in the README and the user guide SHALL build custom systems from `skcomplex.core`, and SHALL reach ready-made
systems only through `catalog`, `describe`, `create_system`, and `create_simulation` of `skcomplex.systems`.

#### Scenario: Import from a field module

- **WHEN** a Python block of the README or the user guide imports from `skcomplex.systems.<field>`
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A field module in the prose

- **WHEN** the prose of the README or the user guide names `skcomplex.systems.<field>`
- **THEN** the documentation test fails and names the file and the line

#### Scenario: A custom system from the core alone

- **WHEN** a reader runs the first custom system of the README in a fresh interpreter
- **THEN** it simulates, and no module of `skcomplex.systems.<field>` is loaded

### Requirement: A user can find compatible components

The user guide SHALL explain the order of choices for a system: time and space, elements, interactions, and solver. It SHALL
show how to read what an interaction needs and what an element carries, in code and on the compatibility page.

#### Scenario: Asking before building

- **WHEN** a reader follows the guide to choose an interaction for an element
- **THEN** the guide's code reads the interaction's requirements and the element's states, and shows the two agree

#### Scenario: Getting it wrong

- **WHEN** a reader follows the guide's example that binds an interaction to an element without the state it reads
- **THEN** the guide shows the report that names the role, the state, and the fix

### Requirement: The user guide is task-led
The user guide SHALL be ordered by what a modeler does: select a ready-made system, change it, build a custom system,
simulate it, record and observe it, analyse it, run experiments, and extend the library. Every current topic SHALL keep a
section. Each section SHALL open with what the reader achieves, then show short code, in the style of the README.

#### Scenario: Reading in order
- **WHEN** a reader follows the sections of the guide from the top
- **THEN** each section builds on the ones before it, and none needs a section that comes later

#### Scenario: No topic lost
- **WHEN** a reviewer compares the topics of the guide before and after the rewrite
- **THEN** every earlier topic has a section

### Requirement: The README shows code only
The Python blocks of the README SHALL hold code only, with no `print` and no output written by hand, on GitHub and on
the documentation site. Blocks that build on earlier ones SHALL be marked as their continuation and render on the site.

#### Scenario: Continuation block on the site
- **WHEN** the documentation site renders a README block marked `python continuation`
- **THEN** it shows as highlighted Python code

### Requirement: The guide shows the values of its code
The documentation build SHALL run the Python blocks of the user guide and show the value of each top-level expression
below its block, with no `print` in the source. Strings SHALL show plain, and `None` and configurable objects SHALL show
nothing. A block that raises SHALL stop the build and name the block.

#### Scenario: Shown value
- **WHEN** a guide block ends with the expression `SIRSystem().get_tags()`
- **THEN** the built page shows the tags below the block

#### Scenario: Failing block
- **WHEN** a guide block raises an error during the documentation build
- **THEN** the build fails and names the page and the block

### Requirement: The prose is plain and lasting
The prose of the README, the guide, the docstrings, and the error messages SHALL use no semicolons and SHALL state no
hard-coded count of the catalog's systems, which goes stale. It SHALL say "in any field of science".

#### Scenario: Count in the prose
- **WHEN** a reviewer reads the README and the guide
- **THEN** no sentence states how many systems the catalog holds, except where generated from the catalog at build time

### Requirement: The documentation follows the scikit-learn split
The README, the user guide, every docstring example, and every gallery example SHALL use `simulate`, the analysis
functions, and the experiment functions. A migration page SHALL list every removed name with its replacement.

#### Scenario: A gallery example
- **WHEN** a gallery example runs a system
- **THEN** it calls `simulate` and the analysis functions, and no `Simulation` or `*Analysis` class

### Requirement: Compatibility is documented per kind of part from tags
The documentation SHALL hold one page per kind of part, which are elements, interactions, laws, solvers, backends,
analyses, and experiments. Each page SHALL be generated from the tags, with a table of what each class declares or
accepts and worked, executed examples of a compatible and an incompatible combination with the error it raises. The
single compatibility page SHALL be removed.

#### Scenario: Choosing a solver
- **WHEN** a reader opens the solvers page
- **THEN** it shows, for each solver, the tags it supports, and an example where a mismatched solver raises an error
  naming the tag

### Requirement: The documentation explains the system hierarchy
The documentation SHALL hold a systems page that explains `BaseSystem`, the composite `System`, general systems of the
core, and field realizations, with an executed example of a subclass that changes a ready-made system. The catalog
pages SHALL show, for each system, its class, its import path, its model parameters, and its capability tags.

#### Scenario: A catalog page
- **WHEN** a reader opens the page of `SIRSystem`
- **THEN** it shows `from skcomplex.systems.biology.epidemiology import SIRSystem`, the parameters, and the tags
