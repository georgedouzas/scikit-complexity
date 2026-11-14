## ADDED Requirements

### Requirement: Docstrings follow the scikit-learn layout
Every docstring of a public class or function of the core SHALL hold, in order, one summary line, one paragraph in plain
words with the formula when there is one, a line that links its section of the user guide, the arguments, a See Also
section, and an Examples section.

#### Scenario: The gradient interaction
- **WHEN** a reader reads the docstring of `GradientInteraction`
- **THEN** its body is one paragraph that gives `dx/dt = -dU/dx` and uses from several fields, and its See Also names
  `Interaction` and `HamiltonianInteraction`

### Requirement: Docstring examples are short and use real parts
The Examples section of a docstring of the core SHALL hold three to six lines of code and SHALL define no class only to
have something to work on.

#### Scenario: A missing example part
- **WHEN** an example of a core docstring defines a class that is not the subject of the example
- **THEN** the test session fails and names the docstring

## MODIFIED Requirements

### Requirement: A docstring of a cross-domain component names uses from several fields
The body paragraph of every docstring of a public class and function of the core components, the analysis functions,
and the experiment functions SHALL say in plain words what the thing does, and SHALL name concrete uses from at least two
fields of science in one sentence.

#### Scenario: The declared element
- **WHEN** a reader reads the docstring of `DeclaredElement`
- **THEN** one sentence names uses such as people infected in an epidemic and moles of a chemical species

#### Scenario: An analysis function
- **WHEN** a reader reads the docstring of `recurrence`
- **THEN** it says what a recurrence is in plain words and names uses from at least two fields

## REMOVED Requirements

### Requirement: A docstring of a core component compares it with its nearest components
**Reason**: The fixed "Unlike", "Choose it when", and "It appears in" paragraphs made every docstring long and alike.
**Migration**: Name the nearest components in the See Also section, one line each, with what tells them apart.
