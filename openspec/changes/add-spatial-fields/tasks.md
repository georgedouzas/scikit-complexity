# Tasks

## 1. Fields

- [ ] 1.1 Add `Field` on lattices and networks, and verify the uniform decay scenario
- [ ] 1.2 Add the operators with stencil and graph-Laplacian lowering on NumPy and Numba, and verify the heat kernel and
  network diffusion scenarios
- [ ] 1.3 Add boundary conditions, and verify the zero-flux scenario
- [ ] 1.4 Add the stability check, and verify the too-large-step scenario

## 2. Coupling and visualization

- [ ] 2.1 Add field reads and deposits for agents through co-simulation inputs, and verify the pheromone trail scenario
- [ ] 2.2 Log fields to Rerun as images, and verify the Gray-Scott scenario on an in-memory recording

## 3. Library

- [ ] 3.1 Add the field models as registered factories, and verify the Fisher front and Turing scenarios and the
  conformance suite

## 4. Documentation and gate

- [ ] 4.1 Document fields, operators, boundaries, and coupling in the user guide, and verify the documentation test
  passes
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
