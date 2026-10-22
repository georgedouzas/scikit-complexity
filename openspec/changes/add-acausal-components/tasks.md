# Tasks

## 1. Ports, equations, and connections

- [ ] 1.1 Add `port` and `PortDomain` to `core.components` and algebraic relations to `Equation` in `core.laws`, and
  verify the resistor ports and same-component scenarios
- [ ] 1.2 Add `connect` to `core.interactions` as an interaction over members with default zero flows, and verify the
  Kirchhoff scenario
- [ ] 1.3 Declare `Requires` for ports, connections, and algebraic equations and `Supports` for the compile stage, add
  the issues `port-time`, `port-domain`, and `structural-singularity` to `validate()`, and verify the domain mismatch,
  wrong time, and high-index scenarios

## 2. Compilation

- [ ] 2.1 Add matching, block sorting, and symbolic elimination to rate laws with algebraic named expressions in
  `core.compile`, and verify the same-component and Kirchhoff scenarios
- [ ] 2.2 Add numeric solution of nonlinear algebraic blocks, and verify the RLC scenario

## 3. Library

- [ ] 3.1 Add the named port domains and the component library to `skcomplex.models.engineering.realizations`, and
  verify the core field-term and import-graph tests still pass
- [ ] 3.2 Add the engineering models as registered factories with their catalog observables, and verify the Chua
  scenario and the conformance suite

## 4. Documentation and gate

- [ ] 4.1 Document ports, equations, connections, validation, and compilation in the user guide, and verify the
  documentation test passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
