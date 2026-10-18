# Design

## Context

Rate laws give explicit ODEs, interactions bind elements through roles, and Sage solves systems of equations
symbolically. Acausal modelling needs a compile stage that turns implicit equations into explicit derivatives.

## Goals / Non-Goals

**Goals:**

- Assemble circuits and mechanisms from reusable components and connections, as Modelica does.
- Reuse every existing solver, backend, analysis, and visualisation through compilation to ODEs.

**Non-Goals:**

- Index reduction of high-index systems by the Pantelides algorithm with dummy derivatives. The compiler reports such
  systems, and a later change can add it.
- Reading Modelica source files.

## Decisions

### Ports and variables

`Port(domain)` declares a connector with the domain's potential and flow variables, each a symbol of the component, so
a resistor has ports `p` and `n` with `p.v`, `p.i`, `n.v`, and `n.i`. `PortDomain(name, potential, flow)` defines new
domains.

### Equations

`Equation(lhs, rhs)` is a law of a component, such as `Equation(p.v - n.v, R * p.i)` and
`Equation(C * derivative(v), p.i)`. `derivative(x)` marks the state variables of the system. A system may mix
equation laws with ordinary rate laws.

### Connections

`connect(a.p, b.n, c.p)` on a system adds the equalities of potentials and the zero sum of flows for the listed ports.
It rejects ports of different domains with a named error. Unconnected flows default to zero, as Modelica does.

### Compilation

The compiler collects the equations and unknowns, and assigns each equation to one unknown with a bipartite maximum
matching. Unknowns are the derivatives of states plus the algebraic variables. It then orders the equations into
strongly connected blocks with Tarjan's algorithm and solves each block symbolically with Sage. Linear blocks are
solved exactly, and nonlinear blocks numerically per evaluation. The result is a system of rate laws with algebraic
variables as observables. A structurally singular system, where some derivative cannot be matched, raises a named error
that lists the unmatched variables and equations, which is how high-index systems are reported.

## Risks / Trade-offs

- [Nonlinear algebraic loops are slow] → They are solved by Newton's method in compiled code with symbolic Jacobians,
  and simple components avoid them.
- [High-index systems are rejected] → The error names the constraint, and the user guide shows how to reformulate
  with a compliant joint.
