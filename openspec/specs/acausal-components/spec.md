# acausal-components Specification

## Purpose

Lets a user assemble systems from elements with typed ports and equations, connect them so that potentials agree and
flows balance, and simulate the result with the existing solvers.

## Requirements

### Requirement: Algebraic states are determined by equations
`StateSpec(algebraic=True)` SHALL declare a state that no law integrates and whose value at every time is a solution of
the equations it appears in. A law that integrates an algebraic state SHALL raise a named error, and the trajectory
SHALL record every algebraic state at every frame.

#### Scenario: Integrated algebraic state
- **WHEN** a rate law targets an algebraic state
- **THEN** compiling raises a named error that names the state and says that its equations determine it

#### Scenario: Recorded voltage
- **WHEN** a voltage divider of two resistors driven by a constant source runs
- **THEN** the trajectory holds the voltage of the middle node at every frame, equal to the divided source voltage

### Requirement: Ports carry potential and flow variables
`ConnectorSpec(name, potential, flow)` SHALL name the variables of a port. An element type SHALL declare its ports in
the class variable `ports`, a mapping from port name to connector, with one algebraic state per port variable named
`<port>_<variable>`. The core SHALL hold no named connector.

#### Scenario: User-defined connector
- **WHEN** a user defines `ConnectorSpec('chemical', potential='mu', flow='n')` and an element type with a port `a` of
  it and the algebraic states `a_mu` and `a_n`
- **THEN** a connection reads the potential `a_mu` and the flow `a_n` of the element, with no change to the core

### Requirement: Components state equations without a causal direction
The equations of an element SHALL relate its states, its port variables, and the derivatives of its states without
assigning which variable each equation determines, and the compiler SHALL solve them for the highest derivatives and
the algebraic states of the system.

#### Scenario: Same component, different use
- **WHEN** a resistor is driven once by a voltage source and once by a current source
- **THEN** both systems compile, solving the resistor's equation for the current in one and for the voltage in the other

### Requirement: Connections balance flows
`ConnectionInteraction` SHALL bind the elements of one node through a variadic role, name the port of each, and return
equation laws that make the potentials of the ports equal and their flows sum to zero. A port that no connection names
SHALL have a zero flow.

#### Scenario: Kirchhoff's current law
- **WHEN** three resistors meet at one node
- **THEN** the currents into the node sum to zero at every frame

### Requirement: Port components exert forces on Lagrangian coordinates
An `EquationLaw` whose target is a coordinate of an `EulerLagrangeInteraction` SHALL add to the equation of that
coordinate, so a component with ports drives a mechanism.

#### Scenario: Motor torque
- **WHEN** a constant voltage drives a DC motor whose torque law targets the angle of an inertia with viscous friction
- **THEN** the angular velocity settles at the value the motor constant, the resistance, and the friction give in
  closed form, within 1%

### Requirement: Connected systems are checked before compilation
`PassCompiler` SHALL report, with the other issues of a system, ports of different connectors in one connection
(`port-connector`), a port in more than one connection (`port-connections`), a declared port without its algebraic
states (`port-states`), and equations that cannot be matched to unknowns (`structural-singularity`), each with its path
and fix.

#### Scenario: Connector mismatch
- **WHEN** a user connects an electrical port to a thermal port
- **THEN** the report holds an issue with the code `port-connector` that names both ports and their connectors

#### Scenario: Shared port
- **WHEN** one port appears in two connections
- **THEN** the report holds an issue with the code `port-connections` whose fix is to list the ports of the node in one
  connection

#### Scenario: Missing port state
- **WHEN** an element type declares a port `p` of the electrical connector without the state `p_i`
- **THEN** the report holds an issue with the code `port-states` that names `p_i`

#### Scenario: High index
- **WHEN** two capacitors are connected in parallel with no resistance between them
- **THEN** the report holds an issue with the code `structural-singularity` that names the constrained capacitor
  voltages, before compilation

### Requirement: Connected systems compile to explicit derivatives
The compiler SHALL match the equations of a connected system to its unknowns, order them into blocks, solve linear
blocks symbolically and nonlinear blocks numerically at every evaluation, so the system runs on every solver of its
form.

#### Scenario: RLC resonance
- **WHEN** a series RLC circuit of components runs from a charged capacitor with small resistance
- **THEN** the capacitor voltage oscillates at a frequency within 1% of `1 / (2π sqrt(L C))`

#### Scenario: Diode current
- **WHEN** a voltage source drives a resistor in series with an exponential diode
- **THEN** the current equals the solution of the diode equation within 1e-8 at every frame
