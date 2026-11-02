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

## Implementation Notes

- A lattice group is an ordinary group with a `space`, created by `Element.lattice`, so its states and parameters use
  the existing layout with one row per site, and parameters per site or shared work unchanged. Element types can keep
  structural settings, such as a Wolfram number or `vmax`, as plain dataclass fields that their laws read from `self`.
- Lattice and Monte Carlo laws bypass the symbolic kernels. The compiler turns them into steps on the problem, in
  declaration order, and two new forms, `LATTICE` and `MONTE_CARLO`, select the lattice stepper and the Monte Carlo
  stepper. Partitions co-simulate them with other forms as before.
- Neighbours come from a precomputed table of member indices per site and offset, with the number of sites standing
  for the fill value outside fixed edges. Periodic, fixed, and reflecting edges are therefore one code path.
- Tables, Wolfram numbers, and totalistic rules become lookup arrays at compile time. On the Numba backend,
  synchronous lookup rules run as one compiled loop: a 1000 × 1000 Game of Life reaches about 220 steps per second,
  and about 55 while streaming images to Rerun. Images use a 256-level palette lookup, since interpolating colors per
  pixel limited the frame rate.
- Lattice forms lower their remaining laws with NumPy, like the agent form. The requested backend only decides whether
  lookup rules are fused.
- `LocalRule` targets can be a tuple that a function rule updates together from the same configuration, which
  Langton's ant needs for its colors and its heading. Function rules receive the component's parameters by keyword
  when they name them.
- `Relaxation` takes an optional `activity` mask. Avalanche sizes count active sites, such as toppling sites, rather
  than every site that changes, which matches the literature's definition of size.
- Monte Carlo energy changes replace `neighbor` calls by placeholders before substituting the proposal, so the
  neighbours keep their values. They are printed to NumPy once. Checkerboard sweeps check that the neighbourhood
  connects the two sublattices only, and suggest the random sequential schedule otherwise.
- The sandpile scenario measures the exponent as the slope of the log-binned density between sizes 1 and 100, giving
  about 1.1. A maximum-likelihood fit of a sample cut at 100 is biased upward by the cut. The full scenario of 100,000
  grains takes about 75 seconds with a generic Python toppling rule, the slowest test of the suite.
- Synchronous majority voting settles into domains that either freeze or blink with period two, so its test accepts
  both.
