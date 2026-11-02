## MODIFIED Requirements

### Requirement: Sage expressions are accepted
Laws, energies, pair potentials, and boundaries SHALL be written as Sage symbolic expressions, without any optional
installation, and the symbols the framework gives a component SHALL be Sage symbols that carry the domains of their
variables. A NumPy scalar given as a law value SHALL become a Sage number, and a law value that is a NumPy array SHALL
raise a named validation error that names the law's target.

#### Scenario: Sage force
- **WHEN** a constant force is given as a Sage expression
- **THEN** the system compiles and simulates with the trajectory of the same force given as a number

#### Scenario: Symbols are Sage symbols
- **WHEN** a component's `laws` method receives its symbols
- **THEN** every scalar symbol is a Sage symbolic expression, and every vector state is a Sage vector with one
  component per dimension of the space

#### Scenario: NumPy scalar times a vector
- **WHEN** a rate law of a vector state `v` has the value `np.float64(2.0) * v * v`, which NumPy builds as an
  element-wise array
- **THEN** compiling raises a named validation error that names the state and asks to convert the NumPy scalar with
  `float`, instead of running the element-wise product

#### Scenario: NumPy scalar as a value
- **WHEN** a rate law has the value `np.float64(-0.5) * x` for a scalar state `x`, or the value `np.float64(1.0)`
- **THEN** the system compiles and its trajectory equals that of the same law written with Python floats
