# Tasks

## 1. Ports, equations, and connections

- [ ] 1.1 Add `Port`, `PortDomain`, the built-in domains, `Equation`, and `derivative`, and verify the resistor ports
  scenario
- [ ] 1.2 Add `connect` with domain checks and default zero flows, and verify the domain mismatch scenario

## 2. Compilation

- [ ] 2.1 Add matching, block sorting, and symbolic elimination to ODEs with algebraic observables, and verify the
  same-component and Kirchhoff scenarios
- [ ] 2.2 Add numeric solution of nonlinear algebraic blocks and the structural singularity error, and verify the RLC
  and high-index scenarios

## 3. Library

- [ ] 3.1 Add the component library and the engineering models as registered factories, and verify the Chua scenario
  and the conformance suite

## 4. Documentation and gate

- [ ] 4.1 Document ports, equations, connections, and compilation in the user guide, and verify the documentation test
  passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
