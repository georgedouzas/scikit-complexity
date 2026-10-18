# Design

## Context

States live in one float vector, and groups compile once whatever their size. `GridSpace` exists for agents on Mesa,
and `PairInteraction` sums pair terms found by neighbour search. Update laws give discrete maps of whole states.

## Goals / Non-Goals

**Goals:**

- Express the general cellular automaton of Ceccherini-Silberstein and Coornaert (2023 def. 1.4.1) on `Z^d` lattices
  and their tori, and run a 1000 × 1000 Game of Life at 30 steps per second on the Numba backend.
- Treat time-scale separation as part of the model, as Charbonneau (2017 §1.4, ch. 5) does.

**Non-Goals:**

- Cellular automata over non-abelian groups and their Cayley graphs, Garden of Eden searches, and subshifts
  (Ceccherini-Silberstein and Coornaert 2023 ch. 5-8), which `add-group-cellular-automata` covers on top of this
  change.
- Cluster Monte Carlo moves (Wolff, Swendsen-Wang). They are recorded for later.

## Decisions

### Lattices and neighbourhoods

`LatticeSpace(shape, boundary='periodic' | 'fixed' | 'reflecting', fill=0)` defines the sites, and
`Neighborhood.moore(r)`, `.von_neumann(r)`, `.radius(r)`, and `Neighborhood(offsets)` define ordered offsets, with the
cell itself as offset zero when included. `Element.lattice(label, space, **values)` creates a cell group with one member
per site, with values given as arrays of the lattice shape.

### Local rules

`LocalRule(target, neighborhood, rule)` is a law. `rule` is one of these:
- a function of the ordered neighbourhood values, which the lowering applies to whole shifted arrays;
- a `Table` mapping patterns to values;
- `Wolfram(number)`;
- `Totalistic(counts_to_value)` or `OuterTotalistic(center, counts_to_value)`.

Tables and totalistic rules lower to integer indexing into a lookup array, and functions lower through the Sage printer
when they are symbolic. Otherwise they run as NumPy functions on arrays. Neighbour values come from `np.roll` on
periodic lattices and padded slices on the others. The Numba backend fuses the stencil into one loop.

### Schedules

`schedule='synchronous'` reads the previous configuration everywhere, as Charbonneau (2017 ch. 2) requires to avoid
directional bias. `'random-sequential'` and `'random-site'` update sites one at a time in random order, and
`'checkerboard'` updates the two sublattices in turn, as the Monte Carlo literature uses. A stochastic rule receives a
uniform random array per step.

### Relaxation

`Relaxation(drive=[laws], relax=[laws], stable=condition, max_iterations)` runs the drive once per step and the relax
laws until `stable` holds, the stop-and-go scheme of Charbonneau (2017 ch. 5). Every relaxation emits an event with
its size (number of updates), duration (iterations), and area (distinct sites), which trajectories store and the event
statistics of `add-analysis-toolkit` read.

### Monte Carlo

`Energy(expression)` declares a site's local energy over its neighbourhood, and `Move.flip()`, `Move.random_value()`,
and `Move.rotate(width)` propose changes. `Sampler('metropolis' | 'glauber' | 'heat-bath')` accepts with the matching
probability at the parameter `temperature`. Sage derives the energy change of a move symbolically from the local
energy. Checkerboard updates make the sweep vectorisable for nearest-neighbour energies.

## Risks / Trade-offs

- [Rules as Python functions are slow] → Tables, Wolfram numbers, and totalistic rules cover the library, and
  functions run on whole arrays, not per cell.
- [New forms add combinations] → Lattice and Monte Carlo partitions co-simulate with ODE and agent partitions through
  the existing scheduler, for example agents on a lattice field.
