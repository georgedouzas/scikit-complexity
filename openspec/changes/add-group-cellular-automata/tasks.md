# Tasks

## 1. Group spaces

- [ ] 1.1 Add `GroupSpace` with quotients, balls, named group constructors, and neighbour tables for local rules, and
  verify the free group and orientation scenarios
- [ ] 1.2 Add the optional `groups` extra with `passagemath-groups` and `passagemath-gap`, lazy imports with a
  suppression that names the extra, and the error that names it, and verify the missing extra scenario
- [ ] 1.3 Add `automaton_group`, `grigorchuk_group`, `permutation_group`, and `matrix_group`, with balls built by
  breadth-first search on the word problem, tree-level hashes resolved by AutomGrp, level quotients, and
  `ball_sizes()`, and verify the Grigorchuk growth, level quotient, free matrix group, and permutation group scenarios
- [ ] 1.4 Add the Cayley graph visual for group spaces, and verify the rules without GAP and Grigorchuk viewer
  scenarios

## 2. Theory

- [ ] 2.1 Add injectivity, surjectivity, and pre-injectivity checks with enumeration and SAT, Garden of Eden search,
  inverse search, and composition, and verify the Life and reversible rule scenarios
- [ ] 2.2 Add `LinearAutomaton` with algebraic decisions, and verify the Laplacian scenario
- [ ] 2.3 Add `Subshift` with checks, counts, and sampling, and verify the golden mean scenario

## 3. Information and library

- [ ] 3.1 Add configuration entropy, entropy rate tracking, and difference patterns, and verify the entropy scenario
- [ ] 3.2 Add the group automata models as registered factories, and verify the Q2R scenario and the conformance suite
- [ ] 3.3 Add `grigorchuk_automaton` under `computer_science/automata`, move its inventory entry from out of scope to
  this change and then to implemented, and verify the Grigorchuk automaton scenario and the inventory check
- [ ] 3.4 Add the gallery example of `grigorchuk_automaton` with its Rerun thumbnail and the ball growth plot, and
  verify the Grigorchuk gallery plot scenario and the gallery budget and coverage checks

## 4. Documentation and gate

- [ ] 4.1 Add PySAT to the dependencies, document group spaces, the `groups` extra, automaton, permutation, and matrix
  groups, theory checks, linear automata, and subshifts in the user guide, and verify the documentation test passes
  with the extra installed in the docs environment
- [ ] 4.2 Run formatting, checks, the docs build, and tests through nox, and verify all of them pass
