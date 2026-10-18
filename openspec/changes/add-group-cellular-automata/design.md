# Design

## Context

`add-lattice-dynamics` defines lattices, ordered neighbourhoods, local rules, and vectorised stencils. A Cayley graph
of a finite quotient is a lattice whose neighbours are given by right multiplication by the memory set. Sage handles
finitely presented groups, permutation groups, and group algebras.

## Goals / Non-Goals

**Goals:**

- Run any automaton of Ceccherini-Silberstein and Coornaert (2023 ch. 1) whose universe has a computable finite
  quotient or a finite ball, with the same rules and schedules as lattices.
- Decide the properties the book studies wherever they are decidable on finite universes.

**Non-Goals:**

- Decision procedures for infinite universes, which are undecidable in general beyond dimension one. The checks work
  on periodic configurations over finite quotients and on finite balls, and the documentation states that limit.
- Sofic and surjunctivity theory beyond examples.

## Decisions

### Group spaces

`GroupSpace(group, quotient=None, ball=None)` takes a Sage finitely presented group or a named constructor such as
`free_group(2)` or `heisenberg()`. With a finite quotient, given as a homomorphism to a permutation group or as a
normal subgroup of finite index, sites are the quotient's elements. With a ball, sites are the words up to the given
length, reduced by Knuth-Bendix or coset enumeration where Sage supports it, and sites on the edge use a fixed fill.
The neighbour table maps each site and memory element `s` to the site `g s`, and becomes the index array of the stencil
lowering.

### Theory checks

On a finite quotient, injectivity and surjectivity of the induced map on configurations are decided by enumeration
for small alphabets and sizes, and by a SAT encoding with PySAT otherwise. Pre-injectivity is checked on pairs of
configurations that differ inside a finite window. Garden of Eden patterns are searched as unsatisfiable preimage
constraints over a window, and `inverse` searches for a local rule with a memory set up to a given size whose
composition is the identity.

### Linear automata

`LinearAutomaton(group, field, coefficients)` represents `Σ a_s s` in the group algebra `K[G]`. Invertibility,
surjectivity, and pre-injectivity follow from the algebra on finite quotients and, for `Z^d`, from Laurent polynomial
arithmetic over the field.

### Subshifts

`Subshift(alphabet, forbidden_patterns)` checks configurations, counts admissible patterns in windows, and samples
admissible configurations by Markov chain moves that respect the constraints.

### Information

`configuration_entropy(subshift or automaton image, windows)` estimates `log |X_F| / |F|` along a Følner sequence of
windows. `difference_pattern(automaton, configuration, site)` returns the space-time set where two runs differ after a
single-cell flip, as Lindgren (2024 ch. 4) uses.

## Risks / Trade-offs

- [Exponential checks] → Enumeration has size limits that raise a named error, and SAT encodings extend them.
- [Knuth-Bendix may not terminate] → Balls require a group with a solvable word problem in Sage, otherwise a named
  error asks for a finite quotient.
