# Design

## Context

After `redesign-system-model`, a system is its components, interactions, space, and time. Rate laws give explicit ODEs,
`Equation` states implicit relations, interactions bind components through roles with the `'members'` binding, and Sage
solves systems of equations symbolically. The redesign's inventory gate maps acausal ports to interactions over members
with algebraic equations, so no new system part is needed. Acausal modelling still needs ports on components and a
compile stage that turns connected implicit equations into explicit derivatives.

## Goals / Non-Goals

**Goals:**

- Assemble circuits and mechanisms from reusable components and connections, as Modelica does.
- Reuse every existing solver, backend, analysis, and visualisation through compilation to ODEs.
- Keep the core application agnostic: ports and domains are general, and the named domains and components live in the
  library.

**Non-Goals:**

- Index reduction of high-index systems by the Pantelides algorithm with dummy derivatives. Validation reports such
  systems, and a later change can add it.
- Reading Modelica source files.

## Decisions

### Ports and variables

`port(domain)` in `core.components` declares a connector on an `Element` with the domain's potential and flow
variables, each a state of the element, so a resistor has ports `p` and `n` with `p.v`, `p.i`, `n.v`, and `n.i`.
`PortDomain(name, potential, flow)` in `core.components` defines a domain. The core ships no named domain: the
electrical, translational, rotational, thermal, and hydraulic domains are realizations in
`skcomplex.models.engineering.realizations`, which the engineering model packages import.

### Equations

`Equation(lhs, rhs)` in `core.laws` and `derivative(x, j)` already exist. A component states relations such as
`Equation(p.v - n.v, R * p.i)` and `Equation(C * derivative(v, 1), p.i)`. This change extends equations from relations
in the highest derivatives to algebraic relations between port variables, without choosing the variable each equation
determines. A system may mix equation laws with ordinary rate laws under `ContinuousTime()`.

### Connections

`connect(a.p, b.n, c.p)` in `core.interactions` returns a stateless `Interaction` with `over = 'members'` and one role
per port's component. Its laws are the equalities of the potentials and the zero sum of the flows of the listed ports.
The system holds it in `interactions=` like any interaction. Unconnected flows default to zero, as Modelica does.

### Compilation

A stage of `core.compile` collects the equations and unknowns, and assigns each equation to one unknown with a
bipartite maximum matching. Unknowns are the derivatives of states plus the algebraic variables. It then orders the
equations into strongly connected blocks with Tarjan's algorithm and solves each block symbolically with Sage. Linear
blocks are solved exactly, and nonlinear blocks numerically per evaluation. The result is a set of rate laws, and the
algebraic variables are named expressions that simulations may record as observables. The stage runs after
validation, so the lowered system runs on every solver and backend whose support covers rate laws.

### Declarations and validation

| Primitive | Declaration | Issue when misused |
|---|---|---|
| `port` | `Requires(time={continuous})` | `port-time`: a port in a system under discrete or static time; the fix is `ContinuousTime()` |
| `connect` | `Requires(time={continuous}, bindings={members})` | `port-domain`: ports of different domains, naming both ports and their domains |
| algebraic `Equation` | `Requires(time={continuous})` | `structural-singularity`: equations that cannot be matched to unknowns, naming the unmatched variables and equations; the fix suggests a compliant element between the constrained states |
| the compile stage | `Supports(time={continuous}, laws={equation, rate}, domains={real})` | equation laws on a non-real domain are reported by the existing law-domain check |

The structural check is part of `validate()`, since it needs only the structure of the equations, so a high-index
system is reported with every other issue before compilation.

## Risks / Trade-offs

- [Nonlinear algebraic loops are slow] → They are solved by Newton's method in compiled code with symbolic Jacobians,
  and simple components avoid them.
- [High-index systems are rejected] → The issue names the constraint, and the user guide shows how to reformulate
  with a compliant joint.

## Open Questions

- The redesign lists `Equation` both as a continuous law (implicit) and as a constraint law (algebraic), but
  `LawKind.EQUATION` has one law class, continuous. Should an equation without derivatives count as the constraint
  class, or get its own law kind? This decides whether a purely algebraic connected system may run under static time.
  The same question is open in `add-static-models`.
