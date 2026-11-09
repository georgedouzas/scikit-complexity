## ADDED Requirements

### Requirement: Module constants lead the module under public names
Every module constant of a Python file of the sources, the tests, and the documents, without the generated pages,
SHALL have a public `UPPER_CASE` name and SHALL sit in one block right after the imports, before the type aliases and
the first definition. The block SHALL hold no blank line and no comment, and no constant SHALL carry an attribute
docstring. A value built from a definition of its own module SHALL follow the same rules right after that definition.

#### Scenario: Constant after a class
- **WHEN** a module defines a literal constant below its first class or function
- **THEN** the source test fails and names the file, the line, and the constant

#### Scenario: Private constant
- **WHEN** a module defines a constant whose name begins with an underscore
- **THEN** the source test fails and names the file, the line, and the constant

#### Scenario: Gap in the block
- **WHEN** a blank line, a comment, or a string sits between two constants of a block, or a comment sits right above it
- **THEN** the source test fails and names the file and the line

#### Scenario: Test and example files
- **WHEN** a test module or a gallery example defines a constant after its first function
- **THEN** the source test fails and names the file, the line, and the constant

#### Scenario: Derived value
- **WHEN** a module builds a dispatch table from functions defined in the same module
- **THEN** the table sits right after the last of them under a public name, and the source test passes

#### Scenario: Renamed constants stay off the surface
- **WHEN** a test lists the names every package re-exports, before and after the renames
- **THEN** the two lists are equal

### Requirement: Classes own the functions they read
A module function of the sources that a class reads, directly or through other module functions, in its module or in
a module that imports it, SHALL be a method of one owning class, which the other readers call through. Only
Numba-compiled kernels, helpers that only public functions read, and functions a derived value reads SHALL stay module
functions. A public function SHALL be re-exported by its package, live in a `_utils` module, or be compiled.

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

### Requirement: Methods begin with a verb
Every method of the sources SHALL begin with a verb, as the constitution asks, apart from magic methods, the hooks a
library calls by a fixed name, properties, and names a base class of another library defines. A helper moved into a
class whose name begins with a noun SHALL take a verb name.

#### Scenario: Moved helper with a noun name
- **WHEN** the private function `_noise` of a reaction-diffusion module becomes a method of its system class
- **THEN** it is named `_get_noise`, and every call site uses the new name

#### Scenario: Method that begins with a noun
- **WHEN** a class of the sources defines a method whose first word is not a known verb
- **THEN** the source test fails and names the file, the line, and the method

#### Scenario: Library hook
- **WHEN** a symbolic function class defines `_eval_` or `_print_`
- **THEN** the source test passes

### Requirement: Symbolic functions are BuiltinFunction subclasses
Every symbolic function the library defines SHALL be an instance of a private `BuiltinFunction` subclass whose
printing, evaluation, and derivative are methods. Sage's `function` factory SHALL be used only to declare an unknown
function of an equation by its name. Expressions SHALL print and compile as before.

#### Scenario: Function with a print hook
- **WHEN** a reviewer reads how the neighbour reduction prints
- **THEN** it is the `_print_` method of its subclass, and no module function holds it

#### Scenario: Factory call with a hook
- **WHEN** a module of the sources calls Sage's `function` at module level or with a hook keyword
- **THEN** the source test fails and names the file and the line

#### Scenario: Unknown function
- **WHEN** a cosmology declares the scale factor with `function('a')` inside a method
- **THEN** the source test passes and the metric reads it as an unknown

#### Scenario: Printing unchanged
- **WHEN** the printer and compile tests run after the conversion
- **THEN** every expression prints, differentiates, and compiles as before

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
