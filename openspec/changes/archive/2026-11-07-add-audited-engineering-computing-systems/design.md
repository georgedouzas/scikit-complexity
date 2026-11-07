# Design

## Context

The audit lists engineering and computer science models that the core supports. They fit existing patterns: elements
with rate laws, impulses, transfer laws of stocks and flows, local rules on lattices, maps on networks and fields,
convolutions of fields, and agent rules over groups.

## Decisions

- Every new system lives in its own module with the components only it uses, so the module reads as an example
  implementation. The coupled Colpitts oscillators carry their own coupling rather than importing one from physics,
  and the components of RobbyGA move into its module.
- Variants become options rather than near copies: the three plants of minimum-time control are one system with a
  `plant` option, the expanded universe of Lenia is the `channels` and `kernels` options of `lenia`, and Gooding's
  evolution by circumstance is a regime of `robby_ga`.
- Minimum-time control follows Dilão's sampled controller: an impulse at every sample sets a held control state to
  the side of the switching curve the plant is on.
- The memductance of the memristive circuit is `a + 3 b w^2`, the derivative of its characteristic. Trzaska prints
  `a + 3 b w`, but only the derivative reproduces his mixed-mode oscillation `2^11`.
- The electromagnet of the shaker pulls with `k2 tan(alpha i2) / alpha`, whose slope at zero current is the `k2` of
  Trzaska's linearized model, which he reports to match the nonlinear one.
- Lenia is a discrete map, `A <- [A + G(K * A) / T]`, as equation 1 of Chan writes it. Its channels are up to three
  states of one field, since a variadic role cannot bind fields.
- The light-emitting robots, the DNA tube, and the evolution by circumstance are agent rules over a group. The robots
  keep the history of their measured light in the storage of their rule, so any delay works, and predict it by the
  slope of the last step for a negative delay.
- Adleman's graph is the seven-node graph of his experiment as commonly reproduced, whose only Hamiltonian path runs
  from 0 to 6 in order. The tube keeps its size, with a fresh edge oligo in the place of every consumed strand.

## Limits

- A variadic role does not bind field elements, so a map cannot read the states of several fields through one role.
  Lenia holds its channels as states of one field and supports at most three channels.
- With the 50 robots of Exercise 10.4 the arena is so bright that every robot moves near its slowest speed, so the
  spreading under a prediction shows clearly only with fewer robots, which the tests and the example use.
