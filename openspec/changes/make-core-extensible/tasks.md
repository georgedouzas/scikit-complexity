# Tasks

## 1. Hooks

- [x] 1.1 Add `_get_member(namespace, role)` to `BaseInteraction` and use it in every symbolic `_get_laws` that reads a role; add a test per general class that renames its target role and compiles with the trajectory of the plain class.
- [x] 1.2 Add `_get_parameter(name)` to `BaseInteraction`, with the plain-attribute fallback for undeclared parameters, and use it for every parameter read in both forms; drop `probability` from `HerdingInteraction` and verify the meadow tests and a renamed-parameter test per agent class.
- [ ] 1.3 Let every variant argument accept a spec or a callable besides its named values, with checks that keep validating the rest; verify a custom pairing, a custom movement step, a custom matching rule, and the error for an unknown name.
- [ ] 1.4 Add `schedule` to the agent forms of sign adjustment, reinforcement, imitation, and reaction, and `ties` to the best-option rules of movement, imitation, and choice; verify the member schedule and random-tie frequencies.
- [ ] 1.5 Declare every member the agent and model views offer on `AgentViewProtocol` and `ModelViewProtocol`; verify mypy on a subclass that uses them.

## 2. Partner selection

- [ ] 2.1 Add `PartnerSpec(candidates, choice, fallback, capacity)` to `skcomplex.specs` with radius candidates from a number or a state, lowered through `MatchLaw` on grids, networks, and continuous space; verify the within-reach-else-nearest and continuous-space scenarios.
- [ ] 2.2 Accept `PartnerSpec` or a pairing callable in transfers, matching, and groupings, with today's named pairings as shorthands giving identical draws; verify every model using them over 20 seeds.
- [ ] 2.3 Let a transfer settle both partners from one shared draw through a settlement callable; verify the meal scenario.
- [ ] 2.4 Rebuild the obesity `EaterElement` on the partner spec, the settlement, periodic phases, and movement; verify its tests over 20 seeds.

## 3. Callables

- [ ] 3.1 Accept callables for crossing guards and resets, jump changes, transition matrices, renewal samplers, reduction functions, and observables of non-agent systems, and `FluxLaw(speed=...)` for Rusanov; verify each against its symbolic form on NumPy, and on Numba with the Python-fallback warning.
- [ ] 3.2 Add `relative`, `random`, and `vectorized` to `CallSpec`; verify the portable element, reproducible stochastic policy, and batch scenarios.
- [ ] 3.3 Make equations, quantum generators and dissipators, and Godunov fluxes raise an error naming them as symbolic only and the callable alternative; update the user guide's callable section; verify the errors.

## 4. Object states

- [ ] 4.1 Let agent states hold hashable objects: 1-D object snapshots, compositions by value, histories of any value, and analyses refusing object states by name; verify the queue scenario.
- [ ] 4.2 Give agent snapshot fields explicit roles in the snapshot spec instead of name guesses; verify an agent with a state named `phase` that is not a heading.

## 5. Errors and records

- [ ] 5.1 Raise clear errors for: a callable Stratonovich amplitude without a derivative (or accept `derivative=`), a non-default activation on update-law phases, grid-only pairings, groupings, movements, and placements in continuous space, and callable values on growing hypergraphs; verify each error message names the hook.
- [ ] 5.2 Fix the silent results: transitions over a cell with collection or callable sources, matching defaults for missing or misspelled states, and agent imitation ignoring `layer`; verify each with a test that failed before.
- [ ] 5.3 Record every remaining restriction with its reason in the specs or design (event tie order, the draw registry, growing-hypergraph limits, closed field boundaries and discretizations, all-default group constructors), and correct the 2.10 statement in `complete-general-model-rewrites`'s archived design about roles in both forms; verify `openspec validate`.

## 6. Documentation and gate

- [ ] 6.1 Document the hooks, partner spec, callable options, and object states in the user guide with runnable examples, and add the changelog entry; verify `pytest --markdown-docs docs/overview`.
- [ ] 6.2 Run formatting, checks, the docs build, and the full suite through nox; verify they pass apart from the known timing tests.
