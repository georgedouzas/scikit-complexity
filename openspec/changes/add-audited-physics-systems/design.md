# Design

## Context

The audit lists physics models that the core supports. Most fit existing patterns: elements with rate, equation, or
update laws, rate and map interactions, Euler-Lagrange and Hamiltonian interactions, cascades and local rules on
lattices, groups on the edges of networks and on the cells of complexes, tensor states, declared lags, and boundaries.

## Decisions

- Variants become options of the system the source varies, rather than near copies: the three-disk table of the
  billiards, the horizontal pivot, the Bernoulli map with a delay, the Nodes-Links-Triangles coupling, and the
  topological, vision-cone, and delayed Vicsek alignments.
- Delays of discrete maps use states with lags. The delayed Vicsek particle keeps a fixed memory of 25 headings, so one
  element serves every delay up to the 25 steps of the source, and a negative delay predicts the headings by a
  least-squares line through the last unwrapped headings.
- The self-organized branching process advances its avalanche one generation per step with one binomial draw per
  step, since a draw cannot take another draw as its parameter. The time of the system counts generations.
- The fixed-energy sandpile topples in parallel with a local rule, since a cascade relaxes within one step and never
  ends in the active phase.
- The rigid dimer is written in the centre and the angle of its rod, the small-step limit of the scheme of the source
  that moves both spheres freely and then restores their distance.
- The three-dimensional axis of an anisotropic particle is held as three components with correlated noises and a
  boundary that rescales it to unit length after every step.
- The degree check of the exterior calculus counts a reduction over a whole group as a number, which it is, so a rate
  on links may scale by the order of the projection of the links on the faces. This is the one core change.

## Limits

- The vision cone and the nearest neighbourhood do not combine, since `NearestNeighborhood` has no cone. The
  combination asked by Exercise 8.11 of Argun et al. needs a nearest neighbourhood restricted to a cone.
- Challenge 5.4 of Argun et al., a rigid body in three dimensions with inertia, and the full chiral diffusion tensor of
  a spiral are not built. The chiral particle couples the translation along its axis to its spin only.
