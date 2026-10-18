# Proposal

## Why

Engineering systems are assembled from components connected through ports, as Ryzhov et al. (2021) do in Modelica
for pendulums, impacts, and controlled mechanisms. Each port carries a potential variable, such as voltage, position,
or temperature, and a flow variable, such as current, force, or heat flow. A connection makes potentials equal and
flows sum to zero. Components state equations rather than assignments, so a resistor is `v = R i` whichever variable
is known. This is Kirchhoff's structure. Mobus (2022 §4.3.3) describes the same structure as interfaces with protocols,
and Schwichtenberg (2020 ch. 2) as continuity on networks. Interactions with roles are close to it, but laws today
assign derivatives to states and cannot state algebraic equations.

## What Changes

- Add ports: typed connectors with potential and flow variables for the electrical, translational, rotational,
  thermal, and hydraulic domains, and user-defined port types.
- Add equation laws: components state equations between their variables, including derivatives and algebraic
  relations, without choosing which variable each equation solves.
- Add connections: connecting ports generates equal potentials and flows that sum to zero, and checks that the port
  types match.
- Compile the equations of a connected system to an ODE by matching variables to equations, sorting them into blocks,
  and solving algebraic blocks symbolically. The system then runs on every existing solver and backend. Systems of
  differential index above one are reported with the variables involved.
- Add a standard component library:
  - electrical: resistor, capacitor, inductor, voltage and current sources, ground, diode, operational amplifier;
  - translational and rotational: mass, spring, damper, force source, inertia, gear;
  - thermal: heat capacity, conductor, convection;
  - hydraulic: tank, pipe, pump.
- Add library models: RLC circuit, Chua's circuit, a quarter-car suspension, a DC motor driving an inertia, and a
  heated room with a thermostat.

## Capabilities

### New Capabilities

- `acausal-components`: ports, equation laws, connections, and their compilation to ODEs.

### Modified Capabilities

- `model-library`: acausal component library and engineering models.

## Impact

- `skcomplex.model` gains `Port`, `Equation`, `connect`, and the port domains, and `skcomplex.compile` gains the
  matching, block sorting, and symbolic elimination stage.
- Depends on `add-general-model-interface`, on `add-higher-order-equations` for `Equation` and `derivative`, and on
  `add-optimal-control` for the thermostat controller in the heated room.
