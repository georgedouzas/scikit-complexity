## ADDED Requirements

### Requirement: Module constants lead the module under public names
Every module constant of `src/skcomplex` SHALL have a public `UPPER_CASE` name and SHALL sit in one block right after
the imports, before the type aliases and the first definition. The block SHALL hold no blank line and no comment, and
no constant SHALL carry an attribute docstring. A value built from a class or function of its own module SHALL follow
the same rules in one block right after the last definition it reads.

#### Scenario: Constant after a class
- **WHEN** a module defines a literal constant below its first class or function
- **THEN** the source test fails and names the module, the line, and the constant

#### Scenario: Private constant
- **WHEN** a module defines a constant whose name begins with an underscore
- **THEN** the source test fails and names the module, the line, and the constant

#### Scenario: Gap in the block
- **WHEN** a blank line, a comment, or a string sits between two constants of a block, or a comment sits right above it
- **THEN** the source test fails and names the module and the line

#### Scenario: Derived value
- **WHEN** a module builds a Sage symbolic function with a print hook defined in the same module
- **THEN** the value sits right after the hook under a public name, and the source test passes

#### Scenario: Renamed constants stay off the surface
- **WHEN** a test lists the names every package re-exports, before and after the renames
- **THEN** the two lists are equal

### Requirement: Classes own the functions they read
A module function that a class reads, directly or through other module functions, in its module or in a module that
imports it, SHALL be a method of one owning class, which the other readers call through. Only three kinds of module
functions SHALL stay: Numba-compiled kernels, helpers of public functions that no class reads, and hooks that a
derived value reads. A public function SHALL be re-exported by its package, live in a `_utils` module, or be compiled.

#### Scenario: Helper of one class
- **WHEN** a private module function is read only by one class of its module
- **THEN** the source test fails and names the function and the class that should own it

#### Scenario: Helper of several classes
- **WHEN** two system classes of one module read the same private module function
- **THEN** the source test fails, and after the fix one class owns it as a static method and the other calls it through
  the owner

#### Scenario: Helper imported by a class of another module
- **WHEN** a pass class imports a private function of another module of its package
- **THEN** the source test fails and names the function, the importing module, and the reading class

#### Scenario: Numba kernel
- **WHEN** a module defines Numba-compiled functions that call one another
- **THEN** they stay module functions and the source test passes

#### Scenario: Helper of an analysis function
- **WHEN** a private function of an analysis module is read only by the public analysis function and its private helpers
- **THEN** it stays a private module function and the source test passes

#### Scenario: Unexported public helper
- **WHEN** a private module defines a public function that no package re-exports and that is not compiled and not in a
  `_utils` module
- **THEN** the source test fails and names the function

#### Scenario: Behaviour after the move
- **WHEN** the test suite runs after the helpers move into classes
- **THEN** every test that passed before passes, with only imports of moved private names changed

## MODIFIED Requirements

### Requirement: Estimators for what is defined, functions for what is done
The public API SHALL hold estimators for the parts a modeler defines and for calibration, and public functions for
simulating, analysing, and running experiments. Every other helper SHALL be a method of the class that reads it, with
only the module functions that the requirement on classes owning the functions they read lets stay. Compilers and
problem forms SHALL be private, with one documented hook for a system package that brings its own form.

#### Scenario: Public surface
- **WHEN** a test lists the public names of `skcomplex.core.engine`
- **THEN** it finds solvers, backends, observables, `simulate`, and `Trajectory`, no concrete compiler or `Simulation`,
  and of compilation and problems only the bases a system package's own form builds on, the problem of the shared pass
  core, and the frames observers read

#### Scenario: A system package brings its own form
- **WHEN** a system class tunes the run setting `compiler` with a compiler its package defines on `BaseCompiler`
- **THEN** `simulate` compiles the system with it, while the library's own forms, such as the rewriting and operator
  forms, are chosen from what the system holds
