# Design

## Context

Automata on groups run as `LocalRuleLaw` laws on `CayleySpace` sites. A memory set is a tuple of words, and the space
turns it into a fixed table of the site of `g s` for every site `g` and word `s`. The compiled `RuleKernel` gathers the
values of the read states through that table and calls the rule. Lattice states hold one number per site, and only
drives change states of several components, such as the matrices of gauge links.

## Goals / Non-Goals

**Goals:**

- Read a group space at sites that depend on the configuration, `x(g x(g) s)`, on finite groups and on balls.
- Give states the alphabet `K[[t]]` truncated at an order, with ring arithmetic.
- Let function local rules read and set states of several components.
- Add the missing CC and EX models as catalog systems.

**Non-Goals:**

- Unbounded memory on an infinite group. The simulation truncates to a ball or a finite quotient.
- Exact arithmetic of power series inside the compiled kernels. Coefficients are floats, exact for integers and for
  residues of a prime field, and the Sage ring is the exact reference.
- Deciding injectivity or surjectivity of rules with translated memory with `AutomatonAnalysis`.

## Decisions

- **A neighbourhood, not a rule option.** `StateTranslationNeighborhood(elements)` is a memory set translated by the
  value of a cell. It names the mathematics, the right translation by `x(g)`, keeps rules as plain functions, and lets
  `LocalRuleLaw.run` and the compiled kernel share the convention. The first read state holds the group elements, as the
  number of their sites, so no state name enters the space layer.
- **A space hook.** `BaseSpace._translation_table(neighborhood)` joins `_index_table` and returns the site of `g h s`
  for every pair of sites and word. `CayleySpace` builds it from its own index tables, walking the shortest word of `h`
  and then `s`, so a ball keeps its boundary convention: a product reached through a site outside reads `fill`. The
  table has `size^2` entries, limited by `max_sites^2`. Other spaces raise a `SpaceError`.
- **Dynamic table in the kernel.** `RuleKernel` takes the translation table and, at every step, picks the row of every
  site at the value of the first read state. The static table it keeps is the translation table flattened, which gives
  the right reading matrix for partial updates. Degrees and colorings depend on a fixed table, so the compiler rejects
  them with a translated memory.
- **Power series as a spec domain.** `PowerSeriesSpec(order, field, variable)` follows `SubshiftSpec` and
  `MatrixGroupSpec`: a state declares `StateSpec(domain=PowerSeriesSpec(...))`, the layout gives it `order` components,
  and entries check every coefficient against `alphabet`, the field or the residues of a prime field. The spec holds
  the truncated Cauchy product, the reduction modulo the characteristic, and conversions to Sage's `PowerSeriesRing`,
  which does not fit NumPy kernels as objects.
- **Components in local rules.** The compiler lets function local rules target states of several components. The
  kernel gathers one row of components per neighbour and writes rows back, for every schedule. Lookup rules stay
  scalar.
- **Systems reuse existing primitives.** Pairs `H x H` and `K^2` are two scalar states set together. Group alphabets
  are numbers of group elements with Sage's Cayley table. Subshift samplers flip cells in random sequential order and
  reject proposals whose ring breaks the constraint, read through a neighbourhood of the whole ring. The Morse subshift
  is the Thue-Morse substitution as a rewriting system. The adding machine adds 1 with carry through a fixed boundary
  that reads the digit `q - 1`.

## Risks / Trade-offs

- The translation table grows as the square of the number of sites. Balls of a few thousand sites fit, and larger ones
  raise `SpaceTooLargeError` with a hint.
- Coefficients over the rationals are floats, so long runs with large coefficients lose exactness. The tests use small
  integers, and `field='GF(p)'` keeps them exact.
- The nilpotent automaton of EX Exercise 1.15 states the memory set `{-1, ..., n - 2}`, whose class is `n - 1`. The
  solution's witness needs `{-1, ..., n - 1}`, which gives class `n`, so the system uses that set and says why.
