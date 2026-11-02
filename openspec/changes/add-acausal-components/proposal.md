# Proposal

## Why

Engineering systems are assembled from components connected through ports, as Ryzhov et al. (2021) do in Modelica
for pendulums, impacts, and controlled mechanisms. Each port carries a potential variable, such as voltage or
temperature, and a flow variable, such as current or heat flow. A connection makes potentials equal and flows sum to
zero. Components state equations rather than assignments, so a resistor is `v = R i` whichever variable is known. This
is Kirchhoff's structure. Mobus (2022 §4.3.3) describes the same structure as interfaces with protocols, and
Schwichtenberg (2020 ch. 2) as continuity on networks.

The current core already provides part of it:
- `EquationLaw` takes a `target`, and `OrdersPass` adds the equations of one target from every element and interaction
  before it solves them jointly for the highest derivatives, so terms from several parts add.
- `EulerLagrangeInteraction` names every coordinate as the target of its equations, so mechanisms compose by summing
  Lagrangians, dissipations, and forcings, and `engineering/common/mechanics/SpringDamperInteraction` joins two masses.
- `CrossingInteraction` switches states when a guard crosses zero, which is a thermostat.
- `rlc_circuit` and `chua` in `engineering/circuits` are registered as lumped equations of their circuits.
- The `law-time` issue of `PassCompiler` already reports an `EquationLaw` under discrete time.

So this change no longer adds translational and rotational port domains, mechanical components, a hydraulic domain no
model uses, a `port-time` issue, the RLC and Chua models, or a dependency on `add-optimal-control`. What the core still
lacks is a variable that no law integrates and that equations alone determine, ports that carry such variables, a
connection interaction, and a solve of equations whose unknowns are not all highest derivatives.

## What Changes

- Add algebraic states: `StateSpec(algebraic=True)` declares a state that no law integrates, whose value at every time
  the equations of the system determine. Its initial value is the starting guess of a nonlinear solve.
- Add ports: `ConnectorSpec(name, potential, flow)` in `skcomplex.specs` names the variables a port carries, and an
  element type declares its ports in the class variable `ports`, a mapping from port name to connector, with one
  algebraic state per variable, named `<port>_<variable>`.
- Add `ConnectionInteraction` in `core.interactions`: it binds the elements of one node through a variadic role, names
  the port of each, and returns `EquationLaw`s that make the potentials equal and the flows sum to zero.
- Extend `OrdersPass`: its unknowns are the highest derivatives and the algebraic states. It matches equations to
  unknowns, orders them into blocks, and solves each block, linear blocks symbolically as it does today and nonlinear
  blocks numerically at every evaluation. The trajectory records every algebraic state.
- Report misuse in `PassCompiler._issues` with the issues `port-connector`, `port-connections`, `port-states`, and
  `structural-singularity`.
- Add the electrical connector and components to `skcomplex.models.engineering.common.electrical`, the thermal
  connector and components to `skcomplex.models.engineering.control`, and the new subfield package
  `skcomplex.models.engineering.mechatronics`.
- Add the models `heated_room` in `engineering/control`, and `dc_motor` and `quarter_car` in
  `engineering/mechatronics`. The motor joins an electrical circuit to a Lagrangian inertia through an `EquationLaw`
  whose target is the inertia's angle. The quarter car is built on `SpringDamperInteraction` and needs no port.

## Capabilities

### New Capabilities

- `acausal-components`: algebraic states, ports, connections, their validation, and the solve of connected equations.

### Modified Capabilities

- `model-library`: the electrical and thermal components and three engineering models.

## Impact

- `skcomplex.specs`: `StateSpec` gains `algebraic`, and `ConnectorSpec` is new.
- `skcomplex.core.elements`: `Element` gains the class variable `ports`, empty by default.
- `skcomplex.core.interactions`: `ConnectionInteraction` is new.
- `skcomplex.engine.compile`: `OrdersPass` solves for algebraic states, and `PassCompiler` reports the port issues.
- `skcomplex.models.engineering`: `common/electrical`, the thermal components of `control`, and `mechatronics`, with
  `pyproject.toml` listing `mechatronics` among the independent subfield packages.
- `add-static-models` solves `EquationLaw`s by their targets under `StaticTime`, in its form `FormKind.STATIC` with
  `RootSolver`. This change solves equations inside the right side under continuous time, so the two do not overlap.
