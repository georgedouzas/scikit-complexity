## ADDED Requirements

### Requirement: A docstring summary names the kind of thing plainly
The summary line of every docstring of `skcomplex.core` SHALL be one line of at most 80 characters. The summary of a
public class SHALL start with "A" or "An" and name its kind, such as element, interaction, law, space, neighbourhood,
system, solver, spec, result, or error. The summary of a function SHALL be imperative. No summary line SHALL hold a term
of a scientific field from the list of the field-term test.

#### Scenario: A class summary
- **WHEN** a reader reads the summary line of `NonnegativeElement`
- **THEN** it reads "An element that holds one nonnegative number."

#### Scenario: A summary that hides the kind
- **WHEN** the summary line of a public class of the core does not start with an article or does not name its kind
- **THEN** the test session fails and names the class and its summary line

#### Scenario: A field term in a summary
- **WHEN** the summary line of a public class or function of the core holds a field term such as "Species"
- **THEN** the test session fails and names the object and the term

### Requirement: A docstring of a cross-domain part names uses from several fields
The docstring body of every public class and function of the core parts, the analysis functions, and the experiment
functions SHALL explain in plain words what the thing does, and SHALL then name concrete uses from at least two fields
of science. Where a use fits as code, the `Examples` section SHALL show it as a runnable example.

#### Scenario: The nonnegative element
- **WHEN** a reader reads the docstring of `NonnegativeElement`
- **THEN** it names uses such as people infected in an epidemic, moles of a chemical species, money in an account,
  and tokens in a place of a Petri net, and its examples build at least two of them and run

#### Scenario: An analysis function
- **WHEN** a reader reads the docstring of `recurrence`
- **THEN** it says what a recurrence is in plain words and names uses from at least two fields

### Requirement: Arguments are described by meaning and units
Every entry of an `Args` section of the core SHALL say what the value means, with its units, range, or allowed values
where they apply, and SHALL NOT describe how the code stores or processes it.

#### Scenario: An argument with units
- **WHEN** a reader reads the argument `x` of `NonnegativeElement`
- **THEN** it says the value is the initial amount in `[0, oo)`, in the units of the system such as people or moles

### Requirement: Jargon is defined where it appears
A docstring or a document SHALL use a term of one field or of the library's internals only with a plain definition in
the same docstring or page, and SHALL call `prepare` and `run` the main public methods, never verbs.

#### Scenario: An undefined term
- **WHEN** a docstring names a class axis or an attractor
- **THEN** the same docstring says what it is in plain words

#### Scenario: The word verb
- **WHEN** a docstring of the core holds the word "verb" or "verbs"
- **THEN** the test session fails and names the object

### Requirement: Module docstrings name their concept in one plain line
The docstring of a private module of the core SHALL be one imperative line that names its concept in plain words. The
docstring of a package `__init__` SHALL be one imperative line, followed by at most one short paragraph that names the
kinds of thing the package holds.

#### Scenario: A private module
- **WHEN** a reader opens the module that defines `NonnegativeElement`
- **THEN** its docstring is the single line "Define the element that holds one nonnegative number."

### Requirement: Documents show every concept in runnable code
The README, the user guide, the generated parts and systems pages, and the gallery examples SHALL show every concept
they introduce with a concrete code block that runs in the test session or the documentation build. The prose SHALL be
short, precise, academic, and simple. It SHALL NOT be informal and SHALL NOT pile up terms that it does not define.

#### Scenario: A concept before its code
- **WHEN** a guide page introduces a distribution over the values of a trait
- **THEN** a runnable block that builds one follows the first paragraph that names it

#### Scenario: A semicolon in a document
- **WHEN** the prose of the README or of a user guide page outside its code blocks holds a semicolon
- **THEN** the test session fails and names the file and the line

### Requirement: The test session guards the mechanical rules of docstrings
The test session SHALL check every module, public class, public function, and public method of `skcomplex.core` for a
summary line longer than 80 characters and for a semicolon in the prose of its docstring. Lines of doctest code SHALL be
exempt from the semicolon check.

#### Scenario: A long summary
- **WHEN** the summary line of a docstring of the core is longer than 80 characters
- **THEN** the test session fails and names the object and the length

#### Scenario: A semicolon in a docstring
- **WHEN** the prose of a docstring of the core holds a semicolon outside a line of doctest code
- **THEN** the test session fails and names the object
