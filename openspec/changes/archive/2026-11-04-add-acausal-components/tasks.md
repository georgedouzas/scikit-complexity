# Tasks

## 1. Algebraic states, ports, and connections

- [x] 1.1 Add `algebraic` to `StateSpec` and `ConnectorSpec` to `skcomplex.specs`, with doctests, and verify that a law
  integrating an algebraic state raises a named error
- [x] 1.2 Add the class variable `ports` to `Element`, and verify the user-defined connector scenario
- [x] 1.3 Add `ConnectionInteraction` to `core.interactions` with its doctest, and verify the Kirchhoff scenario on
  three resistors written in the test

## 2. Solving the equations

- [x] 2.1 Extend `OrdersPass` with algebraic states as unknowns, the zero flow of unconnected ports, the matching, and
  the block order, solving linear blocks as today, and verify the same-component scenario and that every existing
  equation test still passes
- [x] 2.2 Lower nonlinear blocks to a `CallSpec` running `scipy.optimize.root` with the Sage Jacobian, and verify a
  diode circuit against its closed-form current
- [x] 2.3 Record algebraic states in trajectories, and verify the recorded voltage scenario
- [x] 2.4 Add the issues `port-connector`, `port-connections`, `port-states`, and `structural-singularity` to
  `PassCompiler._issues`, and verify the mismatch, shared port, missing state, and high-index scenarios, and that a port
  under discrete time gives the existing `law-time` issue

## 3. Library

- [x] 3.1 Add `engineering/common/electrical` with the electrical connector, resistor, capacitor, inductor, voltage and
  current sources, ground, diode, operational amplifier, and Chua diode, and verify the RLC scenario and that the
  component Chua circuit follows the trajectory of `chua` within its test tolerance
- [x] 3.2 Add the thermal connector, heat capacity, conductor, convection, and heating resistor to
  `engineering/control`, and `heated_room` with its thermostat on `CrossingInteraction`, and verify the heated room
  scenario
- [x] 3.3 Add the `engineering/mechatronics` package with `dc_motor` and `quarter_car`, list it among the independent
  subfield packages in `pyproject.toml`, and verify the motor and quarter-car scenarios
- [x] 3.4 Set the inventory entries `heated_room`, `dc_motor`, and `quarter_car` to implemented with the tags of the
  systems they build, and verify the inventory check and the field-term test pass

## 4. Documentation and gate

- [x] 4.1 Document algebraic states, ports, connections, the solve, and the issues in the user guide, and verify the
  documentation test passes
- [x] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
