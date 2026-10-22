# Tasks

## 1. Group spaces

- [x] 1.1 Add `GroupSpace` to `skcomplex.core.space` as a lattice-kind space with quotients, balls, and neighbour tables
  for `LocalRule` laws on site groups, and verify the free group and orientation scenarios
- [x] 1.2 Add the declarations of `GroupSpace` and the validation issues for the checkerboard schedule, image visuals,
  and balls above the site limit, and verify the unsupported schedule and site limit scenarios
- [x] 1.3 Add the optional `groups` extra with `passagemath-groups` and `passagemath-gap`, lazy imports with a
  suppression that names the extra, and the error that names it, and verify the missing extra scenario
- [x] 1.4 Add the general constructors `finitely_presented_group`, `automaton_group`, `permutation_group`, and
  `matrix_group` to `skcomplex.core.space`, with balls built by breadth-first search on the word problem, tree-level
  hashes resolved by AutomGrp, level quotients, and `ball_sizes()`, and verify the free matrix group and permutation
  group scenarios
- [x] 1.5 Add the named groups `free_group`, `heisenberg`, `lamplighter`, `baumslag_solitar`, and `grigorchuk_group` to
  `skcomplex.models.computer_science.automata` as realizations over the general constructors, and verify the
  Grigorchuk growth and level quotient scenarios and that the core names no group
- [x] 1.6 Add the Cayley graph visual for group spaces to `skcomplex.core.viz`, and verify the rules without GAP and
  Grigorchuk viewer scenarios

## 2. Theory

- [x] 2.1 Add injectivity, surjectivity, and pre-injectivity checks with enumeration and SAT, Garden of Eden search,
  inverse search, and composition to `skcomplex.core.analysis.automata`, and verify the Life and reversible rule
  scenarios
- [x] 2.2 Add `LinearAutomaton` with algebraic decisions, and verify the Laplacian scenario
- [x] 2.3 Add `Subshift` with checks, counts, and sampling, and verify the golden mean scenario

## 3. Information and library

- [x] 3.1 Add configuration entropy, entropy rate tracking, and difference patterns, and verify the entropy scenario
- [x] 3.2 Add the group automata models as registered factories with their catalog observables, and verify the Q2R
  scenario and the conformance suite
- [x] 3.3 Add `grigorchuk_automaton` under `computer_science/automata`, move its inventory entry from out of scope to
  this change and then to implemented with its vocabulary tags, and verify the Grigorchuk automaton scenario and the
  inventory check
- [ ] 3.4 Add the gallery example of `grigorchuk_automaton` with its Rerun thumbnail and the ball growth plot, and
  verify the Grigorchuk gallery plot scenario and the gallery budget and coverage checks

## 4. Documentation and gate

- [x] 4.1 Add PySAT to the dependencies, document group spaces, the `groups` extra, automaton, permutation, and matrix
  groups, theory checks, linear automata, and subshifts in the user guide, regenerate the compatibility table, and
  verify the documentation test passes with the extra installed in the docs environment
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
