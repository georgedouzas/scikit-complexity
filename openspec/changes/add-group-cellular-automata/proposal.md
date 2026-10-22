# Proposal

## Why

Ceccherini-Silberstein and Coornaert (2023) define cellular automata over any group, not only `Z^d`:
- **General definition:** configurations are maps from a group to an alphabet, and an automaton is a shift-equivariant
  map given by a finite memory set of group elements and a local rule (def. 1.4.1).
- **Garden of Eden theorem:** over amenable groups, an automaton is surjective exactly when it is pre-injective
  (ch. 5). The theorem fails over free groups.
- **Linear automata:** these correspond to elements of group rings (ch. 8).
- **Subshifts of finite type:** these constrain configurations by forbidden patterns, according to their companion book
  of exercises (ch. 1).
- **The Grigorchuk group:** a finitely generated group of intermediate growth, amenable but not finitely presented,
  defined by its action on the rooted binary tree (sec. 6.9). Relators cannot describe it, so its Cayley graph must be
  built from the word problem that the tree action solves.

Lindgren (2024 ch. 4) adds the information-theoretic study of automata, and *The Structures of Mathematical Physics*
(ch. 2) the group structures behind lattice symmetries. `add-lattice-dynamics` covers lattices and their tori. This
change generalises the universe to finitely generated groups, given by finite presentations, by Mealy automata, or as
permutation or matrix groups, and adds the automata theory the books develop.

**Rebased on `redesign-system-model`.** The redesign settled the vocabulary this change builds on: local rules are the
`LocalRule` law with `Table`, `Totalistic`, and `OuterTotalistic` lookups, a lattice is a `LatticeSpace` of kind
`SpaceKind.LATTICE` with an ordered neighbourhood, cell states are a `Group(on='sites')`, the step is
`DiscreteTime()`, observables belong to catalog entries, and rule codes live in the automata models rather than the
core. It does not cover group universes. This change still adds the group space, as a lattice-kind space in
`core.space` with its declarations and validation, the general group constructors, and the automata theory in
`core.analysis`; the named groups (free, Heisenberg, lamplighter, Baumslag-Solitar, Grigorchuk) are realizations and go
to the automata model package instead of the core.

## What Changes

- Add group spaces: Cayley graphs of finitely presented groups, given by generators and relations, built on finite
  quotients or on balls of a given radius. The automata models build free groups, the Heisenberg group, lamplighter
  groups, Baumslag-Solitar groups, and finite groups from the general constructors. Memory sets are given as group
  elements, so neighbourhoods keep their orientation. A group space is a lattice-kind space, so the solvers and backends
  that declare lattice support run it, and validation reports what it cannot offer.
- Add group spaces of finitely generated groups given by other presentations, whose balls are built by solving the word
  problem of the presentation instead of applying relators:
  - automaton (self-similar) groups given by a Mealy automaton, as wreath recursions of their action on a rooted tree,
    such as the Grigorchuk group, with finite quotients given by the action on a level of the tree;
  - permutation groups and matrix groups over the integers or a prime field, given by generators.
- Build every group from GAP through `passagemath-gap`, with its packages AutomGrp and FR for automaton groups, in an
  optional `groups` extra, because GAP is licensed under GPL-2.0-or-later and the project is MIT.
- Stream group spaces to Rerun as Cayley graphs, with node colours from the cell states.
- Run `LocalRule` laws on site groups of group spaces under `DiscreteTime`, with the same rule forms and schedules as
  on a `LatticeSpace`.
- Declare the requirements of group spaces and report through validation the lattice options a group space does not
  offer, such as the checkerboard schedule of a square lattice and image streaming.
- Add automata theory on finite universes:
  - checks of injectivity, surjectivity, and pre-injectivity on periodic configurations;
  - searches for Garden of Eden patterns and for mutually erasable pairs;
  - the inverse rule of a reversible automaton;
  - composition of automata.
- Add linear automata over finite fields and the reals as group-ring elements in Sage, with invertibility and
  surjectivity decided algebraically.
- Add subshifts of finite type: forbidden patterns as constraints, checked on configurations, and used to sample
  admissible configurations.
- Add information measures of automata: the entropy of configuration sets along Følner sequences, the spatial entropy
  rate over time, and difference patterns from single-cell perturbations.
- Add library models:
  - the majority automaton on the free group;
  - the lamplighter automaton;
  - the discrete Laplacian as a linear automaton;
  - the reversible Q2R Ising automaton;
  - the golden mean shift as a subshift;
  - the majority automaton on the Grigorchuk group (`grigorchuk_automaton`), previously recorded out of scope because
    the group is not finitely presented.

## Capabilities

### New Capabilities

- `group-cellular-automata`: group spaces of finitely presented, automaton, permutation, and matrix groups, automata
  theory checks, linear automata, subshifts, and automaton information measures.

### Modified Capabilities

- `model-library`: group automata models, including `grigorchuk_automaton`.

## Impact

- `skcomplex.core.space` gains `GroupSpace` and the general group constructors `finitely_presented_group`,
  `automaton_group`, `permutation_group`, and `matrix_group`. A new module `skcomplex.core.analysis.automata` gains
  `Subshift`, `is_injective`, `is_surjective`, `is_pre_injective`, `garden_of_eden`, `inverse`, `compose`,
  `LinearAutomaton`, `configuration_entropy`, and `difference_pattern`.
- `skcomplex.models.computer_science.automata` gains the named groups `free_group`, `heisenberg`, `lamplighter`,
  `baumslag_solitar`, and `grigorchuk_group` as realizations over the general constructors, next to its rule codes.
- `skcomplex.core.viz` gains a Cayley graph visual for group spaces.
- `pyproject.toml` gains the optional extra `groups = ["passagemath-groups>=10.8.13", "passagemath-gap>=10.8.13"]`,
  and building a group space without it raises an error that names the extra. NumPy remains the only requirement of
  running a rule once the space is built.
- `docs/overview/inventory.toml` moves `grigorchuk_automaton` from out of scope to pending this change.
- Depends on `redesign-system-model` for `LocalRule`, site groups, the space declarations, and validation, and on
  `add-analysis-toolkit` for block entropies.
