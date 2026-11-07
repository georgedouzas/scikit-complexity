## Context

The reference audit (reports A to E) checked every catalog entry against the PDFs of its sources. Some entries cite a
section that does not hold their rules, some systems implement a different model than the chapter they cite, and some
matches were never recorded.

## Goals / Non-Goals

**Goals:**
- Every implemented entry cites the place its rules come from.
- Docstrings and registered references say which model a system implements.

**Non-Goals:**
- Implementing the cited models that are missing (Caiani's guru network, Bellomo's continuous crowd, Chesney's Ajani's
  Choice and Fair Food Program, Gooding's Toy Trader and double auction, Gros's dHAN). Other changes add them.
- Changing the behaviour of any system.

## Decisions

- **Primary paper over a loose book match.** When a system's rules are not in the cited chapter, the entry cites the
  original paper of its rules (Lux and Marchesi 1999, Brock and Durlauf 2001, Epstein and Axtell 1996, Gode and Sunder
  1993, Fukai and Tanaka 1997) as an author-year citation, which the inventory check accepts. Where a book section holds
  part of the rules, such as the order-driven market of Ca 3.2.1 or the kinetic network structure of Be 3.4, it is cited
  for that part only.
- **Problems are cited by number.** Argun's Nicholson-Bailey and food webs are problems of section 14.6, cited as
  `Ar Problem 14.7`, like the existing `Rm Ex` citations.
- **The unbounded-alphabet automaton cites the rule it lifts.** CC Example 3.3.3 in the 2023 edition is
  `max(0, x(n) - x(n + 1))`, and the rule `x(n - 1) + x(n)` appears nowhere in CC or its exercises. Its reduction modulo
  2 is the mirror image of CC Example 3.3.8, so the entry cites that example, and the docstring says so.
- **The exercises book gets its own key `EX`**, since it shares authors and year with CC.
- **arXiv matches** are added only where the paper writes the model the system implements. The generic activator-inhibitor
  Turing system of 2105.13722 is matched to `schnakenberg`, not to `gray_scott`, whose substrate-depletion kinetics the
  paper does not use.

## Risks / Trade-offs

- [Titles change for a few entries] → the titles are display strings only, used by the gallery scaffold.
