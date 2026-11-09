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

### Requirement: A docstring of a core component compares it with its nearest parts
The docstring of every public class of the core elements, interactions, laws, spaces, neighbourhoods, times, and
systems SHALL hold a paragraph that starts with "Unlike" and names what the class adds over its nearest part, followed
by a sentence that starts with "Choose it when", and a paragraph of uses that starts with "It appears in".

#### Scenario: The nonnegative element compared
- **WHEN** a reader reads the docstring of `NonnegativeElement`
- **THEN** it says that, unlike `Element`, it needs no subclass and keeps its one state at zero or above, and when to
  choose it

#### Scenario: A missing comparison
- **WHEN** a public class of a rewritten family has no paragraph that starts with "Unlike" or "It appears in"
- **THEN** the test session fails and names the class

### Requirement: The guide is a reference of the core components for builders
The user guide SHALL hold one hand-written reference page per family: elements, interactions, laws, spaces with their
neighbourhoods and times, and systems. Every public class SHALL have a section that says what it models, how it
combines with other parts, its limits and the errors it raises, and how to subclass it into a field realization, with
runnable examples from at least two fields that use the class unchanged.

#### Scenario: The elements page
- **WHEN** a reader opens the elements page of the core components reference
- **THEN** the section of `NonnegativeElement` simulates an epidemic and a bank account with the same element and
  `RateInteraction`, shows an error it raises with the message and the fix, and subclasses it into a field realization

#### Scenario: The examples run
- **WHEN** an example of a reference page raises an error it does not show or prints a wrong value
- **THEN** the test session fails and names the page and the block

#### Scenario: A class without a section
- **WHEN** a public class of a family has no section on its page
- **THEN** the test session fails and names the class and the page

### Requirement: Each family has generated fact sheets and a comparison table
The parts page of each family SHALL be generated from the code. It SHALL hold a comparison table with one row per
public class, giving what it adds, when to choose it, and its typical uses from its docstring, and a fact sheet per
class with its states, roles, parameters, law kinds, the times, solvers, and backends that accept it from its tags,
and links to its reference section, its API entry, and the catalog systems that use or subclass it.

#### Scenario: The fact sheet of an element
- **WHEN** a reader opens the parts page of elements
- **THEN** the fact sheet of `NonnegativeElement` lists its state `x` with its domain, the solvers that accept it, and
  links to systems such as `SIRSystem` that use it or its field realizations

#### Scenario: A new class
- **WHEN** a contributor adds a public element class with the comparison and uses paragraphs
- **THEN** the table and the fact sheets show it with no edit of the generator

### Requirement: The guide shows how the parts fit together
The reference SHALL open with a page that explains how elements, interactions, laws, spaces, times, and systems fit
together, with diagrams of their relations and of the run of a system from `prepare` to a trajectory, and a worked
example that builds one system step by step, running the code after every step.

#### Scenario: The diagrams render
- **WHEN** the documentation builds the page on how the parts fit
- **THEN** its diagrams render as figures and the build reports no error

#### Scenario: The worked example
- **WHEN** a reader runs the blocks of the worked example in order
- **THEN** each block runs, and the last one simulates the finished system

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

### Requirement: The prose reads as natural human writing
Docstrings and documents SHALL use no semicolons and no dashes as punctuation: no em dash, no en dash, and no hyphen
between spaces. Hyphenated words, list markers, and code stay allowed. They SHALL NOT use stock openers, closers, or
filler words from a banned list. They SHALL vary sentence length, prefer precise verbs, use a paragraph where a list
adds nothing, and back every claim with code or a reference.

#### Scenario: A dash in prose
- **WHEN** the prose of a docstring of the core, the README, a user guide page, or a gallery example holds an em dash,
  an en dash, or a hyphen between spaces outside code
- **THEN** the test session fails and names the file or the object and the line

#### Scenario: A banned phrase
- **WHEN** such prose holds a banned phrase such as "it is worth noting", "delve", "leverage", "seamless", "robust",
  or "powerful" outside code and the titles of cited references
- **THEN** the test session fails and names the file or the object and the phrase

### Requirement: The guide teaches the workflows of a system end to end
The user guide SHALL hold one tutorial per workflow: build, check, simulate, analyse, evaluate, calibrate, experiment,
visualise, and share. Each tutorial SHALL follow one running example, show every public function or method of its
workflow with runnable code, and explain each result it prints.

#### Scenario: The calibrate tutorial
- **WHEN** a reader follows the calibrate tutorial
- **THEN** it fits the running example to data with every method of `calibrate`, prints the estimates and the
  diagnostics of each, and says when to choose each method

#### Scenario: The check tutorial
- **WHEN** a reader follows the check tutorial
- **THEN** it prepares the running example, reads its tags, and triggers the common validation errors, showing each
  message and the fix

### Requirement: Case studies run the whole workflow in several fields
The user guide SHALL hold at least two complete case studies from different fields. Each SHALL take one system through
building, checking, simulating, analysing, evaluating or calibrating, experimenting, and visualising, with data or a
known result to compare against, and every block SHALL run in the test session.

#### Scenario: An epidemic calibrated to case data
- **WHEN** a reader follows the epidemic case study
- **THEN** it builds an SIR system, calibrates it to published outbreak counts, checks the fit and the posterior, and
  runs scenarios of the calibrated system

#### Scenario: A phase transition
- **WHEN** a reader follows the lattice case study
- **THEN** it sweeps a spin lattice over temperature and locates the transition near the known critical value

### Requirement: The guide holds a glossary, a decision guide, and a troubleshooting page
The user guide SHALL hold a glossary that defines every mathematical and library term the guide uses, a decision guide
that leads from the features of a problem to the core parts that fit it, and a troubleshooting page with one section
per public error class of the core that shows code raising it, its message, and the fix.

#### Scenario: An error without a section
- **WHEN** a public error class of `skcomplex.core.errors` has no section on the troubleshooting page
- **THEN** the test session fails and names the class

#### Scenario: Choosing a part
- **WHEN** a reader starts the decision guide with a population that lives on a network and changes in discrete time
- **THEN** the guide leads to `GroupElement` on a `NetworkSpace` with `DiscreteTime` and links their reference sections

### Requirement: Docstrings, guide, and catalog link to each other
Every reference section and tutorial SHALL link each core class it names to its API entry, every fact sheet SHALL link
the reference section and the catalog systems of its class, and every catalog system page SHALL link the core classes
it builds from. The documentation build SHALL fail on a broken link.

#### Scenario: From the catalog to the core
- **WHEN** a reader opens the catalog page of `SIRSystem`
- **THEN** it links the core classes it builds from, each to its fact sheet

### Requirement: The test session guards the mechanical rules of docstrings
The test session SHALL check every module, public class, public function, and public method of `skcomplex.core` for a
summary line longer than 80 characters, and for a semicolon, a dash used as punctuation, or a banned phrase in the prose
of its docstring. Lines of doctest code and inline code SHALL be exempt.

#### Scenario: A long summary
- **WHEN** the summary line of a docstring of the core is longer than 80 characters
- **THEN** the test session fails and names the object and the length

#### Scenario: A semicolon in a docstring
- **WHEN** the prose of a docstring of the core holds a semicolon outside a line of doctest code
- **THEN** the test session fails and names the object

## MODIFIED Requirements

### Requirement: The user guide is task-led
The user guide SHALL be ordered by what a modeler does: select a ready-made system, change it, learn the core
components, build a custom system, check it, simulate it, analyse it, evaluate it, calibrate it, run experiments,
visualise it, and share it, followed by case studies and the glossary, decision guide, and troubleshooting pages. Every
current topic SHALL keep a section. Each section SHALL open with what the reader achieves, then show short code, in the
style of the README.

#### Scenario: Reading in order
- **WHEN** a reader follows the sections of the guide from the top
- **THEN** each section builds on the ones before it, and none needs a section that comes later

#### Scenario: No topic lost
- **WHEN** a reviewer compares the topics of the guide before and after the rewrite
- **THEN** every earlier topic has a section
