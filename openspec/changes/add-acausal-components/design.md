# Design

## Context

A system is its elements, interactions, and nested systems. `EquationLaw(lhs, rhs, target=None)` states an implicit
relation, and `OrdersPass` in `skcomplex.engine.compile.passes` turns every equation into a relation in placeholders of
the highest derivatives it holds. It adds the relations of one target, keyed by the state and the column of the target,
checks that each sum holds a highest derivative, and solves all of them jointly: symbolically with Sage up to six
unknowns, and by `numpy.linalg.solve` inside a `CallSpec` above. The unknowns must appear linearly.

`EulerLagrangeInteraction` returns one equation per coordinate with that coordinate as its target, so the Lagrangian of
a coupling adds to the equations of the coordinates it reads. The issues of a system come from `PassCompiler._issues`
as `IssueSpec(code, path, message, fix)` in the `report_` that `prepare(system)` sets, and `EquationLaw` already
requires continuous or static time, which the `law-time` issue reports.

## Goals / Non-Goals

**Goals:**
- Assemble circuits and thermal networks from reusable components and connections, as Modelica does.
- Keep every solver, backend, analysis, and visual working, by solving the equations inside the right side.
- Keep the core free of field terms: algebraic states, connectors, and connections are general, and the named
  connectors and components are field cases in the models.

**Non-Goals:**
- Index reduction of high-index systems by the Pantelides algorithm with dummy derivatives. The structural check
  reports such systems.
- Mechanical ports. A mechanism composes through `EulerLagrangeInteraction`, and a port component that exerts a force
  on a coordinate adds an `EquationLaw` with that coordinate as its target.
- Reading Modelica source files.

## Decisions

### Algebraic states

`StateSpec(algebraic=True)` declares a state that no law integrates. It has order 1 and no lags, a law that integrates
it is an error, and its value at every time is a solution of the equations it appears in. The initial value is the
starting guess of a nonlinear solve. The trajectory records it like any other state, so observables and visuals read
it unchanged.

### Ports and connectors

`ConnectorSpec(name, potential, flow)` in `skcomplex.specs` is a frozen spec of the names of the potential and flow
variables of a port, such as `ConnectorSpec('electrical', potential='v', flow='i')`. An element type declares its ports
in the class variable `ports: ClassVar[Mapping[str, ConnectorSpec]]`, which `Element` sets to an empty mapping, and
declares in its constructor one algebraic state per port variable, named `<port>_<variable>`, so a resistor with ports
`p` and `n` holds `p_v`, `p_i`, `n_v`, and `n_i`. The states are declared like every other state, so no attribute is
generated. A flow is positive into the element.

The core ships no named connector. The electrical connector and components live in
`skcomplex.models.engineering.common.electrical`, which the `control` and `mechatronics` packages share, and the thermal
connector and components live in `skcomplex.models.engineering.control`, their one user.

### Connections

`ConnectionInteraction(label, *, members, ports)` in `core.interactions` binds the elements of one node through the
variadic role `members` and names the port of each element in `ports`, in the same order. Its laws are `EquationLaw`s:
the potential of every port after the first equals that of the first, and the flows of all ports sum to zero. A port
belongs to one connection, and a node with many ports is one connection, as a node of a netlist is. A port that no
connection names has a zero flow, as in Modelica, which the interaction does not need to state: the pass adds the
equation `flow == 0` for it.

### Solving the equations

`OrdersPass` extends its solve without changing what it does for equations in highest derivatives alone:
- The unknowns are the placeholders of the highest derivatives and the symbols of the algebraic states. A relation
  whose sum holds no unknown is a structural singularity.
- It matches relations to unknowns by a maximum bipartite matching, `networkx.bipartite.hopcroft_karp_matching`, and
  orders the matched pairs into blocks, the strongly connected components of their dependency graph in topological
  order, `networkx.condensation` and `networkx.topological_sort`. NetworkX is already a runtime dependency.
- A block linear in its unknowns is solved as today, by Sage up to six unknowns and by a numeric linear solve above.
- A nonlinear block lowers to a `CallSpec` that runs `scipy.optimize.root` with the Jacobian Sage derives, starting
  from the values of the last evaluation, which the NumPy backend runs.
- The algebraic states are substituted by their solutions in the rate laws, and the solutions are recorded as the
  values of the algebraic states at every frame.

No library was adopted for the structural analysis itself, because NetworkX holds the matching and the components, and
the pass only maps relations and unknowns to its graphs. CasADi and the Modelica compilers were rejected because they
would replace the symbolic layer rather than plug into it.

### Field cases on the core

A component is an element type in the models with ports and `EquationLaw`s:
- A resistor is `EquationLaw(s.p_v - s.n_v, s.R * s.p_i)` and `EquationLaw(s.p_i + s.n_i, 0)`.
- A capacitor holds a state `u` of order 1 with `EquationLaw(s.C * derivative(s.u, 1), s.p_i)` and the port
  relations.
- The DC motor is an interaction of `mechatronics` that binds the armature and the inertia. It returns the back
  electromotive force `EquationLaw(s.armature.p_v - s.armature.n_v, s.k * derivative(s.inertia.phi, 1))` and the
  torque `EquationLaw(-s.k * s.armature.p_i, 0, target=s.inertia.phi)`, which adds to the Euler-Lagrange equation of
  the angle that an `EulerLagrangeInteraction` gives the inertia.

The thermostat of `heated_room` is a `CrossingInteraction` with hysteresis that switches the heater state on and off.

### Issues

`PassCompiler._issues` adds four issues:
- `port-connector`: the ports of one connection have different connectors. The issue names each port and its
  connector, and the fix is to connect ports of one connector.
- `port-connections`: a port appears in more than one connection. The issue names the port and the connections, and
  the fix is to list every port of the node in one connection.
- `port-states`: an element type declares a port without an algebraic state for one of its variables. The issue names
  the state, and the fix is to declare `<port>_<variable>` with `StateSpec(algebraic=True)`.
- `structural-singularity`: the matching leaves relations or unknowns unmatched. The issue names them, and the fix is
  to add a compliant element, such as a resistance, between the constrained states.

The structural check needs only which unknowns every relation holds, so it runs in `prepare` with the other issues,
before compilation.

## Risks / Trade-offs

- [Nonlinear blocks are slow] → They are solved with Sage Jacobians from the last values, and simple components avoid
  them.
- [High-index systems are rejected] → The issue names the constraint, and the user guide shows how to reformulate it
  with a compliant element.
- [`add-static-models` also solves algebraic equations] → Both changes use the same algebraic states and the same
  solve, and whichever lands second builds on the first.

## Open Questions

- None.
