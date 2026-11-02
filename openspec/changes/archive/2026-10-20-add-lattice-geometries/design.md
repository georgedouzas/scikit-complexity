# Design

## Context

Every lattice law compiles to one table, `LatticeSpace.index_table(neighborhood)`, with one row per site and one column
per offset. A site outside fixed edges points at the index `size`, which reads the fill value. `CompiledRule`,
`CompiledRelaxation`, and `CompiledMonteCarlo` only gather from this table, so they do not depend on the geometry. Two
things do:
- the sublattices of the checkerboard schedule, which assume even and odd coordinate sums;
- `ImageVisual`, which draws a lattice state as an image of the lattice shape.

Fields already read graph neighbours. `compile/_fields.py` builds a padded neighbour table from a `NetworkSpace` for the
graph Laplacian. The Bak-Sneppen update is not a sweep. It changes one site and its neighbours per application, chosen
by the global minimum.

## Goals / Non-Goals

**Goals:**

- Run every existing lattice law on triangular lattices and on graphs without a second code path in the steppers.
- Keep the triangular lattice vectorised and exact, with ordered directions for lattice gases.
- Make local energies exact on sites of different degree.
- Add the extremal update as a general law, usable for any key state and any neighbourhood.

**Non-Goals:**

- The honeycomb lattice as an offset lattice. It has a two-site basis, so it runs as a graph from
  `networkx.hexagonal_lattice_graph`.
- Neighbours ordered by bearing on graphs. Direction-dependent rules use the triangular or square lattice.
- An indexed priority queue for the extremal search. A vectorised `argmin` over the key state is fast enough at the
  sizes of the references, and the queue would need a mutable heap that NumPy and Numba do not share.
- Graphs that change during a run, which `add-adaptive-networks` covers.

## Decisions

### Library survey

The constitution asks the change to take its geometry from a mature library and to keep only the adapter in house.
The survey, checked in October 2026:

| Library | What it offers | Maintenance and license | Wheels, Python 3.12–3.13 | Verdict |
|---|---|---|---|---|
| NetworkX 3.x | `triangular_lattice_graph`, `hexagonal_lattice_graph`, `grid_2d_graph`, `grid_graph`, all with periodic edges and `pos` attributes, `greedy_color` and `equitable_color`, `to_scipy_sparse_array` | Active, BSD-3, already a runtime dependency (3.7 installed) | Pure Python, all platforms | Wrapped |
| rustworkx 0.17 | Hexagonal and grid lattice generators, coloring, fast traversal in Rust | Active, Apache-2.0 | Linux, macOS | Rejected: a second graph library for generators NetworkX already has, and the hot loops run in NumPy and Numba, not in graph code |
| python-igraph 0.11 | Square lattices, coloring in C | Active, GPL-2.0 | Linux, macOS | Rejected: GPL-2.0, and no triangular or honeycomb generator |
| graph-tool 2.x | Lattice and triangulation generators, fast C++ | Active, LGPL-3.0 | Conda only, no PyPI wheels | Rejected: no wheels |
| Hexagonal grid packages such as `hexy` and `hexalattice` | Axial coordinates and plotting of hexagonal grids | Small, rarely maintained | Pure Python | Rejected: the axial coordinates are a few lines of offsets, and the packages do not compute neighbour tables |
| NetKet, ALPS | Monte Carlo on lattices and graphs | Active NetKet (Apache-2.0, JAX), ALPS unmaintained | NetKet on Linux and macOS | Rejected: quantum variational Monte Carlo with its own model language, which cannot take the framework's symbolic energies |

No library offers an extremal update over a user rule. The search is `numpy.argmin` or `numpy.argmax` over the key
state, with ties broken by the simulation's generator, and the update reuses the compiled rule. The in-house part is
the selection of the sites and the event record.

The adapter therefore holds:
- the axial offsets and the positions of the triangular lattice;
- the conversion of a NetworkX graph to the padded index table, built from `nx.to_scipy_sparse_array` in CSR form;
- the coloring of the neighbour graph by `nx.greedy_color` with the `'largest_first'` strategy, cached per table.

### Triangular geometry on `LatticeSpace`

`LatticeSpace(shape, geometry='square' | 'triangular', ...)` keeps integer coordinates. A triangular site `(i, j)` sits
at `(j + i / 2, i √3 / 2)` in the plane, so every boundary and `index_table` work unchanged. `Neighborhood.triangular(
center=False)` returns the offsets `(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)`, at 0°, 60°, ..., 300°, so
direction `k + 3` is opposite to direction `k`. The property `positions` returns the site positions for both
geometries, scaled by the spacing.

Alternative considered: lattice spaces built from `networkx.triangular_lattice_graph`. It loses the ordered
directions FHP needs, its periodic positions are shifted for drawing, and it gives up the vectorised offset tables.
The graph route stays available for users who want it.

### Lattice groups on `NetworkSpace`

A lattice group accepts a `NetworkSpace`, with one site per node in `graph.nodes` order. `NetworkSpace.index_table(
neighborhood)` returns the padded table of an `Adjacency(center=False)`, with `size` in the padding columns, the same
convention as fixed edges. The field compiler maps the padding to the site itself, as it does today, so fields and
lattice laws share one table. A `capacity` other than `None` is rejected for lattice groups, since sites hold states,
not agents. Initial values are an array in node order or a mapping from node to value.

Alternative considered: a new graph lattice space. It would duplicate the graph that `NetworkSpace` already holds for
agents and fields.

### Degrees and neighbour sums

A function rule that names the keyword `degree` receives the number of real neighbours per site. Lookup rules on graphs
take only the totalistic and outer-totalistic forms, since a table over ordered patterns needs a fixed width.
`neighbor_sum(expression)` is a symbolic function, written with `adjacent(state)` for the value of one neighbour. The
Monte Carlo compiler lowers it to `Σ_k mask_k · expression(neighbour_k)` over the padded columns, with the mask from
the padding index, so a padded slot contributes nothing whatever the expression. On square and triangular lattices the
mask is false only outside fixed edges. The existing `neighbors(state, neighborhood)` stays for ordered energies.

### Colored schedule

`schedule='colored'` on `LocalRule` and `Sampler` partitions the sites by a proper coloring of the graph whose edges are
the non-padding entries of the law's table. NetworkX colors it once per table at compile time, and each color class
updates in one vectorised call. The checkerboard check still rejects a triangular lattice or a graph, and its message
suggests the colored schedule.

### Extremal law

`Extremal(target, neighborhood, rule, *, key=None, extremum='min', repeat=1, label=None)` applies `repeat` times per
step. Each application:
1. finds the extremal site of `key`, the first target by default, with ties broken uniformly by the simulation's
   generator;
2. gathers the site and its real neighbours, without duplicates;
3. calls the rule `f(*values, random, **parameters)` with one array per target in that order and a uniform array of the
   same length;
4. writes the returned arrays back.

Every application appends an `ExtremalRecord(label, time, site, value)` to the trajectory events, next to the
relaxation records. The gap, the running extremum of the recorded values, and the avalanches of Bak-Sneppen are
computed from these records by the existing event analysis.

### Compile pipeline, backends, and runtime

The new laws are declarations in `skcomplex.core.model`, and `skcomplex.core.compile` lowers them:
- `neighbor_sum` energies are Sage expressions, lowered by `numpy_function` like every energy, with one mask symbol
  per column of the table;
- tables, masks, and color classes are NumPy arrays computed once per problem;
- on the Numba backend, an extremal law runs the applications of a step in one compiled loop, generated for its
  targets and parameters, that draws from the same `numpy.random.Generator`, which Numba accepts as an argument, so
  both backends give the same events, and lookup rules use the existing fused kernel, which reads any table. Numba
  0.6x cannot compile functions with keyword-only arguments, so the compiled loop needs a rule that takes its
  arguments by position or keyword, such as `f(fitness, random)`. The compiler checks the signature and compiles the
  loop once at compile time, and an extremal law whose rule Numba does not compile runs the NumPy loop on the Numba
  backend, with the same events.

`LatticeStepper` and `MonteCarlo` run the compiled steps unchanged, and a frame returns the state of a triangular
lattice in its shape and the state of a graph lattice in node order.

### Visualization

`SiteVisual` logs a triangular lattice as `rr.Points2D` at the site positions, with radii that tile hexagonal cells and
colors from the palette `ImageVisual` uses. It logs a graph lattice as `rr.GraphNodes` colored by state, with static
`rr.GraphEdges`, and with node positions from the `pos` attribute when the graph has one. It is registered as
`'sites'` and is the default visual of lattice groups that are not square.

## Risks / Trade-offs

- [Padding wastes memory on graphs with a few hubs] → The table is as wide as the largest degree. Models of the
  references use regular or near-regular graphs, and the user guide names a scale-free graph of 10⁵ nodes as the
  practical limit for the padded table.
- [A greedy coloring uses more colors than needed] → More colors only mean more vectorised calls per sweep. The
  sampling stays correct because no two neighbours change together.
- [Extremal search costs one pass over the lattice per application] → At 10⁴ sites and 10⁶ applications this is about
  10¹⁰ comparisons in the compiled loop, a few seconds on the Numba backend. The scenarios use 200 sites.
- [Ties in the key state on integer lattices] → The generator breaks them, so runs stay reproducible from the seed.
