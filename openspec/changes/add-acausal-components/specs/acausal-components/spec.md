# Spec Delta

## Purpose

Lets a user assemble systems from components with typed ports and equations, connect them so that potentials agree and
flows balance, and simulate the result with the existing solvers.

## ADDED Requirements

### Requirement: Ports carry potential and flow variables
An element SHALL declare ports with `port(domain)`, each with the potential and flow variables of a domain defined by
`PortDomain(name, potential, flow)`. The core SHALL hold no named domain.

#### Scenario: Resistor ports
- **WHEN** a user inspects the ports of the library's resistor
- **THEN** it has two ports of the electrical domain, each with a voltage and a current

#### Scenario: User-defined domain
- **WHEN** a user defines `PortDomain('chemical', potential='mu', flow='n')` and declares a port of it
- **THEN** the port has the states `mu` and `n`, with no change to the core

### Requirement: Components state equations
Components SHALL state equation laws between their variables, their port variables, and the derivatives of their
states, without assigning which variable each equation determines.

#### Scenario: Same component, different use
- **WHEN** a resistor is driven once by a voltage source and once by a current source
- **THEN** both systems compile, solving the resistor's equation for the current in one and for the voltage in the other

### Requirement: Connections balance flows
`connect` SHALL return a stateless interaction with the `'members'` binding whose laws make the potentials of the
connected ports equal and their flows sum to zero.

#### Scenario: Kirchhoff's current law
- **WHEN** three resistors meet at one node
- **THEN** the currents into the node sum to zero at every frame

### Requirement: Ports and connections are validated
Ports, connections, and algebraic equations SHALL declare their requirements, and `validate()` SHALL report, with the
other issues of the system, ports of different domains, ports in a system without continuous time, and structurally
singular equations, each with its path and fix.

#### Scenario: Domain mismatch
- **WHEN** a user connects an electrical port to a thermal port
- **THEN** validation reports an issue with the code `port-domain` that names both ports and their domains

#### Scenario: Wrong time
- **WHEN** a system with ports declares `DiscreteTime()`
- **THEN** validation reports an issue with the code `port-time` whose fix is continuous time

#### Scenario: High index
- **WHEN** two capacitors are connected in parallel with no resistance between them
- **THEN** validation reports an issue with the code `structural-singularity` that names the constrained capacitor
  voltages, before compilation

### Requirement: Connected systems compile to ODEs
The compiler SHALL turn the equations of a validated connected system into rate laws by matching and symbolic
elimination, and SHALL keep the algebraic variables as named expressions that simulations may record as observables.

#### Scenario: RLC resonance
- **WHEN** a series RLC circuit runs from a charged capacitor with small resistance
- **THEN** the capacitor voltage oscillates at a frequency within 1% of `1 / (2π sqrt(L C))`
