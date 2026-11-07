# Proposal

## Why

The reference audit lists the models of Ceccherini-Silberstein and Coornaert, Cellular Automata and Groups (2023, CC)
and of its exercises book (EX) that the catalog lacks. Two of them need capabilities the core does not have. The map
`τ(x)(g) = x(g · x(g))` of CC Example 1.8.2 reads a group space at a site that depends on the value of the cell, so
no fixed memory set describes it. The automaton `τ(x)(n) = x(n) − t x(n + 1)` of CC Exercise 1.37 takes values in the
ring `K[[t]]` of formal power series, an alphabet of infinite dimension. The other missing models fit the existing
local rules, group spaces, and subshift sampling, and only need catalog systems.

## What Changes

- Add `StateTranslationNeighborhood` to `skcomplex.core.components.spaces`, a memory set translated by the group
  element a cell holds, so a local rule reads `x(g x(g) s)`. A group space gives the sites of these products through a
  new hook, on a finite group exactly and on a ball with the fill value outside it, the stated boundary convention of
  the truncation.
- Add `PowerSeriesSpec` to `skcomplex.core.specs`, the ring `K[[t]]` over the rationals, the reals, or a prime field,
  truncated at a declared order. A state whose domain is the spec holds the coefficients of a series as its
  components, with the truncated ring product and conversions to Sage's power series ring.
- Local rules with function rules read and set states of several components per cell, one row of components per cell.
- Add catalog systems in `computer_science/automata` for the CC and EX models: the self-indexed shift, the power series
  automaton, the subtraction automaton over the naturals, Muller's pre-injective non-surjective automaton on F2, the
  surjective non-pre-injective linear automaton on F2, Life on the free group F4, the nilpotent automaton of class n,
  the involutive automaton, the group-valued difference automaton, the real quadratic automaton, the S-gap subshifts
  with the even subshift, the Morse subshift, the context-free subshift, and the adding machine. Each gets tests, an
  inventory entry, and a gallery example.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `group-cellular-automata`: reads at state-dependent group elements, states in truncated power series rings, local
  rules over states of several components, and the CC and EX catalog systems.

## Impact

- Code: `core/components/spaces` (new neighbourhood, Cayley space hook), `core/specs` (new spec, entry domains),
  `core/components/laws/_local_rule.py`, `core/engine/problems/_rule_kernel.py`, `core/engine/compile/passes`
  (lattices and layout), and `systems/computer_science/automata`.
- Docs: inventory entries and gallery examples for the new systems, and the lattice systems guide.
- No new dependency and no new package.
