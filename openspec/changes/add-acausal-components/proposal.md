# Proposal

## Why

Engineering systems are assembled from components connected through ports, as Ryzhov et al. (2021) do in Modelica
for pendulums, impacts, and controlled mechanisms. Each port carries a potential variable, such as voltage, position,
or temperature, and a flow variable, such as current, force, or heat flow. A connection makes potentials equal and
flows sum to zero. Components state equations rather than assignments, so a resistor is `v = R i` whichever variable
is known. This is Kirchhoff's structure. Mobus (2022 §4.3.3) describes the same structure as interfaces with protocols,
and Schwichtenberg (2020 ch. 2) as continuity on networks. Interactions with roles are close to it, but equation laws
today state implicit relations in the highest derivatives only, and cannot state algebraic relations between the
variables of connected components.

**Rebased on `redesign-system-model`.** The redesign already provides the `Equation` law with its law kind, the
four-part `System`, continuous time, and interactions with the `'members'` binding, and its inventory gate maps acausal
ports to interactions over members with algebraic equations. This change still adds ports and port domains as a general
component primitive, connections as interactions whose laws are equations, the algebraic use of `Equation` between port
variables, and the compile stage that turns connected equations into explicit derivatives. The named port domains
(electrical, translational, rotational, thermal, hydraulic) are field vocabulary, so they live in
`skcomplex.models.engineering.realizations`, not in the core.

## What Changes

- Add ports: typed connectors with a potential and a flow variable, declared by a component with `port(domain)`, and
  `PortDomain(name, potential, flow)` to define a domain. The core holds only these general forms.
- Extend equation laws: components state algebraic relations between their variables and port variables, alongside
  relations in derivatives, without choosing which variable each equation solves.
- Add connections: `connect(...)` returns an `Interaction` with the `'members'` binding whose laws are the equalities of
  potentials and the zero sum of flows of the connected ports. Validation reports ports of different domains.
- Compile the equations of a connected system to explicit derivatives by matching variables to equations, sorting them
  into blocks, and solving algebraic blocks symbolically. The system then runs on every existing solver and backend.
  Systems of differential index above one are reported with the variables involved.
- Declare `Requires` for ports and connections and `Supports` for the compile stage, so validation reports misuse
  before compilation.
- Add a library of port domains and components in `skcomplex.models.engineering.realizations`:
  - the electrical, translational, rotational, thermal, and hydraulic port domains;
  - electrical: resistor, capacitor, inductor, voltage and current sources, ground, diode, operational amplifier;
  - translational and rotational: mass, spring, damper, force source, inertia, gear;
  - thermal: heat capacity, conductor, convection;
  - hydraulic: tank, pipe, pump.
- Add library models: RLC circuit, Chua's circuit, a quarter-car suspension, a DC motor driving an inertia, and a
  heated room with a thermostat.

## Capabilities

### New Capabilities

- `acausal-components`: ports, algebraic equation laws, connections, their validation, and their compilation to
  explicit derivatives.

### Modified Capabilities

- `model-library`: the port domains, the acausal component library, and the engineering models.

## Impact

- `skcomplex.core.components` gains `port` and `PortDomain`, `skcomplex.core.interactions` gains `connect`, and
  `skcomplex.core.laws` extends `Equation` to algebraic relations between port variables. `skcomplex.core.compile`
  gains the matching, block sorting, and symbolic elimination stage, and `skcomplex.core.base` the issue codes of ports
  and connections.
- `skcomplex.models.engineering.realizations` gains the port domains and the component library.
- Depends on `redesign-system-model`, and on `add-optimal-control` for the thermostat controller in the heated room.
