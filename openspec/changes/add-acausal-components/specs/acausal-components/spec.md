# Spec Delta

## Purpose

Lets a user assemble systems from components with typed ports and equations, connect them so that potentials agree and
flows balance, and simulate the result with the existing solvers.

## ADDED Requirements

### Requirement: Ports carry potential and flow variables
Components SHALL declare ports of a domain with a potential and a flow variable, for built-in and user-defined domains.

#### Scenario: Resistor ports
- **WHEN** a user inspects the ports of a resistor
- **THEN** it has two electrical ports, each with a voltage and a current

### Requirement: Components state equations
Components SHALL state equations between their variables and the derivatives of their states, without assigning which
variable each equation determines.

#### Scenario: Same component, different use
- **WHEN** a resistor is driven once by a voltage source and once by a current source
- **THEN** both systems compile, solving the resistor's equation for the current in one and for the voltage in the other

### Requirement: Connections balance flows
Connecting ports SHALL make their potentials equal and their flows sum to zero, and SHALL reject ports of different
domains.

#### Scenario: Kirchhoff's current law
- **WHEN** three resistors meet at one node
- **THEN** the currents into the node sum to zero at every frame

#### Scenario: Domain mismatch
- **WHEN** a user connects an electrical port to a thermal port
- **THEN** a named error names both ports and their domains

### Requirement: Connected systems compile to ODEs
The compiler SHALL turn the equations of connected systems into ODEs by matching and symbolic elimination, keep the
algebraic variables as observables, and report structurally singular systems with the variables involved.

#### Scenario: RLC resonance
- **WHEN** a series RLC circuit runs from a charged capacitor with small resistance
- **THEN** the capacitor voltage oscillates at a frequency within 1% of `1 / (2π sqrt(L C))`

#### Scenario: High index
- **WHEN** two capacitors are connected in parallel with no resistance between them
- **THEN** compiling raises a named error that names the constrained capacitor voltages
