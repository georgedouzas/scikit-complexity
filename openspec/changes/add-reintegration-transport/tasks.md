## 1. Reintegration

- [ ] 1.1 Add `reintegrate` and its symbolic operator in `src/skcomplex/core/components/laws/utils/_operators.py`,
  export it, and verify its doctest and the error for its use inside a rate law
- [ ] 1.2 Lower it on the NumPy backend as rolled arrays over the window with the periodic and fixed edges, and verify
  the shift-by-one-site, half-a-site, random-flow, and fixed-edge scenarios in
  `tests/core/components/laws/utils/test_operators.py`
- [ ] 1.3 Lower it in the Numba printer, and verify that the scenarios of 1.2 give the same values on Numba
- [ ] 1.4 Lower it on JAX, and verify the gradient-of-the-travelled-distance scenario
- [ ] 1.5 Verify the two-parameters-meet scenario on every backend
- [ ] 1.6 Document reintegration and carried fields in the section on nonlocal operators of the user guide with a
  runnable example, and verify the markdown tests run it

## 2. Flow-Lenia

- [ ] 2.1 Add `FlowLeniaSystem` and its elements and interactions in
  `src/skcomplex/systems/computer_science/automata/_flow_lenia.py`, export them, and verify the mass-conserved and
  defaults scenarios
- [ ] 2.2 Verify the patterns-stay-localised scenario, marked slow if it exceeds the time budget of the suite
- [ ] 2.3 Add `localized` and `mutation_rate`, and verify the two-species scenario
- [ ] 2.4 Add the inventory entry `flow_lenia` with the source 2212.07906 and the primitive `mass-transport` with
  `change = "add-reintegration-transport"`, and verify the inventory check passes
- [ ] 2.5 Add the gallery example `docs/examples/computer_science/automata/plot_flow_lenia.py`, with a run from a
  random patch and a two-species run with localised weights, and verify the strict gallery build runs it

## 3. Integration

- [ ] 3.1 Commit the work as conventional `feat` commits and verify `nox -s changelog` lists them
- [ ] 3.2 Run `nox` formatting, checks, tests of the touched modules, and the strict docs build, and fix every finding
