# Design

## Context

See proposal.md for the motivation. The current core has:
- `NetworkSpace(graph, layers=, hyperedges=)` in `skcomplex.core.spaces`, which numbers edges in the order of
  `graph.edges`, orients an undirected edge from its first end in node order, and holds `edges_` and `adjacency_`.
- `GroupElement(label, element=, space=, on='sites' | 'edges' | 'hyperedges')` in `skcomplex.core.elements`, one member
  per part of a network space, and `FieldElement` for sites of lattices.
- The law utilities `gradient`, `divergence`, `laplacian`, and `neighbors(expression, over=, reduce=, weight=)` of
  `skcomplex.core.laws.utils`, and the neighbourhoods of `skcomplex.core.spaces` that derive from `BaseNeighborhood`.
- `SamplingInteraction(target, state, energy, move, temperature, rule, schedule, neighborhood, values, width)`, which
  lowers to a `SampleLaw` and runs on `MonteCarloSolver` under `DiscreteTime`. Its target is a group on the sites of a
  lattice or the nodes of a network, and its `'colored'` schedule colors the conflict graph with
  `networkx.greedy_color`.
- `CouplingInteraction(mode='edges')`, whose function reads the member, the neighbour, and the edge between them, and
  `RateInteraction`, a rate on a state of any order.
- Angle states, `StateSpec(period=2 * pi)`, of kind `DomainKind.ANGLE`, and integer states.
- Validation through `PassCompiler.prepare(system).report_`, with the existing codes such as `law-space`,
  `law-domain`, and `solver-unsupported`.

Nothing yet orients the hyperedges of a network as faces, maps between degrees, or samples a state on edges. The
`arbitrage_traders` model of `economics/finance` writes Schwichtenberg's connection as a site parameter `A_x(t)`.

## Goals / Non-Goals

**Goals:**
- Write a lattice gauge model as its link variables, its plaquette energy, and its sampler, without hand-written index
  arithmetic.
- Give the same cell states and exterior calculus to non-physics models, such as exchange rates on a currency graph and
  flows on networks.
- Keep the core free of field terms: the Wilson action, gauge transformations, and gauge observables live in the model
  packages.

**Non-Goals:**
- Non-abelian link variables such as SU(2) and SU(3), with their heat-bath and overrelaxation updates. The four models
  need U(1) and Z_N only, which angle and integer states already hold. A later change can add matrix-valued states.
- Fermions and hybrid Monte Carlo.
- States on cells of degree 3 or more, which take part in incidence and topology only.
- Laws run once per hyperedge of an interaction, which `add-higher-order-interactions` adds. The laws of a group on
  hyperedges already run once per member.

## Library survey

The survey checked PyPI, the repositories, and the licences in October 2026, for an MIT-compatible licence, wheels for
Linux and macOS on Python 3.12 and 3.13, maintenance, and an interface the compiler can lower symbolic laws to.

| Library | What it provides | Verdict |
|---|---|---|
| Grid and GPT | lattice QCD in C++ with a Python front end | rejected: GPL-2.0, no wheels, needs a C++ toolchain |
| Chroma, QDP++ | lattice QCD application suite | rejected: no Python interface |
| `lyncs` | Python bindings to lattice QCD libraries | rejected: 0.0.3 from 2020, source only |
| pyQCD | Wilson action and heat-bath in Cython | rejected: not on PyPI, no U(1) or Z_N, no graphs |
| `lgt` | lattice gauge theory in Python | rejected: no licence, no Python 3.12 |
| `edlgt` | exact diagonalization of quantum gauge Hamiltonians | rejected: quantum Hilbert spaces, not classical links |
| PyDEC | discrete exterior calculus on simplicial and cubical complexes | rejected: not installable from PyPI under its name |
| `fdtd`, MEEP | finite-difference time-domain electromagnetics | rejected: own grid model, no graphs, MEEP is GPL and conda only |
| GUDHI, Ripser | persistent homology of filtrations | rejected: a different task, the Betti numbers of one complex are a rank computation |
| NetworkX | oriented incidence, cycle bases, cliques, greedy coloring | wrapped: network complexes, faces, and coloring |
| SciPy | sparse matrices and sparse least squares | wrapped: incidence products, Hodge decomposition, and ranks |
| Numba | compiled loops | wrapped: gather kernels of the exterior calculus |

No maintained library holds exterior calculus on lattices and graphs with a permissive licence and wheels. The in-house
parts are the index tables of cubical complexes and the lowering of the operators, thin code over NumPy, SciPy,
NetworkX, and Numba, which the project already depends on.

## Decisions

- **Reference numbers are checked before they become thresholds.** The topological Kuramoto gap and the U(1)
  plaquette tolerance are first reproduced from the cited source; when a run disagrees, the scenario uses the measured
  value with a tolerance from the spread over seeds, and design.md records the source, the measured value, and why
  they differ. Decided on 2026-11-01.

### Cell complexes are network spaces

`CellComplexSpace(space, *, faces=None, dimension=2, label='space')` derives from `NetworkSpace`, so every group,
binding, and neighbourhood of a network space applies to it. Its constructor stores its arguments, and `_prepare`
derives the graph, the hyperedges, and the incidence before the network attributes of the parent are set:
- For a `LatticeSpace` of `d` dimensions it numbers the links as `(direction, site)` and the plaquettes as
  `(plane, site)`, with planes ordered `(0, 1), (0, 2), ...`. Fixed and reflecting edges drop the cells that leave the
  box, so the complex has a boundary. Cubes are held for incidence when `dimension` is 3.
- For a `NetworkSpace`, `faces` is `None` for a minimum cycle basis, `'hyperedges'` for the hyperedges of the network,
  `'cliques'` for the clique complex up to `dimension` through `networkx.enumerate_all_cliques`, or a sequence of node
  cycles.

It sets `incidences_`, the oriented incidence matrices between consecutive degrees as SciPy CSR matrices with entries
`±1`, and `cofaces_`, the padded table of the cells of the next degree that contain every cell, which the kernels read.
A face whose node cycle uses a pair the graph does not join raises a `SpaceError` that names the face and the missing
edge, in `_prepare`.

Alternative considered: letting `LatticeSpace` offer edges and hyperedges directly. It would duplicate the orientation
and incidence logic in two spaces, and groups on edges already need a network space.

### Exterior calculus

`exterior_derivative`, `codifferential`, and `hodge_star` are symbolic functions in a module `_cells` of
`skcomplex.core.laws.utils`, like `inverse_laplacian`. The degree of their argument is the part its group lives on, 0
for sites, 1 for edges, and 2 for hyperedges. A law of an element of a group reads its own degree, and a law of an
interaction whose roles bind groups of adjacent degrees reads the other group's states, such as the current on links and
the charge on sites. The compiler lowers the operators to products with `incidences_` on the NumPy backend and to
gather loops over `cofaces_` on the Numba backend. The Hodge star of a lattice complex is diagonal, the ratio of dual to
primal cell volumes from the spacings, and on a network complex it is the diagonal of the edge and face weights given by
`weight` arguments of the edge and hyperedge groups, one by default. The codifferential is `(-1)^k ⋆⁻¹ d ⋆`, so
`δδ = 0` holds exactly as `dd = 0` does. The Hodge Laplacian is `codifferential(exterior_derivative(x)) +
exterior_derivative(codifferential(x))`, so no separate name is added, and `gradient`, `divergence`, and `laplacian`
keep their site meaning.

### Coface reductions

`CofaceNeighborhood()` derives from `BaseNeighborhood`. In `neighbors(expression, over=CofaceNeighborhood())`, the
members of a reduction are the faces that contain an edge, or the edges that contain a site, with their orientation
sign. Inside such a reduction, `exterior_derivative(s.theta)` is the oriented sum of the edge values around the one
face, so the local energy of a link is
`-beta * neighbors(cos(exterior_derivative(s.theta)), over=CofaceNeighborhood())`. For U(1) and Z_N states the face sum
is the holonomy angle, so no holonomy function is needed.

### Sampling edge groups

`SamplingInteraction` accepts a target on the edges of a `CellComplexSpace`. Its energy reduces over
`CofaceNeighborhood()`, its moves are the existing `'rotate'` for U(1) and `'random-value'` with `values=range(n)` for
Z_N, whose `'heat-bath'` rule already draws among all values by their Boltzmann weights. The `'colored'` schedule colors
the graph of edges that share a face with the existing greedy coloring, so lattices of odd size need no special case.
Every draw comes from the simulation's generator, so a seed reproduces a run on both backends.

### Validation

`PassCompiler` reports, with the new code `law-degree`, a `codifferential` of a site state, an `exterior_derivative` of
a hyperedge state on a complex of dimension 2, and a `SamplingInteraction` on a hyperedge group, each with the path of
the law and a fix that names the operation of the right degree. A coface reduction outside a cell complex is reported
with the existing code `law-space`.

### Real-time link fields

Rate laws on edge groups compile like any group under `ContinuousTime`, so `RK4Solver`, `ScipyODESolver`, and
`VelocityVerletSolver` run them. Lattice Maxwell writes `A_t = -E` and `E_t = δ dA - J` with `RateInteraction`s.
Because the codifferential of the electric rate is `δδ dA - δJ`, the Gauss law holds to roundoff for every stepper.

### Analysis

`CellComplexAnalysis(kind, space=None, variables=None)` reads a state of a group of a cell complex from a trajectory, or
an array with the complex passed as `space`:
- `'hodge'` solves two sparse least-squares problems with `scipy.sparse.linalg.lsqr`, for the site and the face
  potentials, and sets `gradient_`, `curl_`, and `harmonic_`, which are orthogonal and sum to the flow.
- `'betti'` sets `betti_` from the ranks of the incidence matrices.
- `'curvature'` sets `share_` and `energy_`, the share of faces whose oriented sum of an angle field is not a multiple
  of its period, and the sum of `1 - cos` of the face sums.

### Visualization

`CellVisual(state, plane=(0, 1), slice=0)` in `skcomplex.engine.viz` logs an edge group as `LineStrips2D` colored by
value, with a cyclic color map for angle states, and a hyperedge group as an image of the lattice shape on a lattice
complex or as colored polygons on a network complex. A lattice of three or four dimensions logs the plane and slice it
selects.

### Models and their field cases

- `lattice_gauge_fields(size=32, *, dimension=2, n=None, beta=2.0, dynamics='monte-carlo', charges=None)` in
  `physics/waves`. With Monte Carlo dynamics, one edge group holds angle links, U(1) when `n` is `None` and Z_N
  otherwise, sampled by `WilsonSamplingInteraction`, a thin subclass of `SamplingInteraction` in the package whose
  energy is the Wilson action. With `dynamics='hamiltonian'`, the edge group holds a real potential and an electric
  field, and a site group holds static charges. `GaugeAnalysis(BaseAnalysis)` in the package measures Wilson loops,
  Polyakov loops, the topological charge, and the Gauss residual, and the catalog entry declares the mean plaquette and
  the Gauss residual as observables.
- `topological_kuramoto` in `physics/synchronization` holds node and link phases on a clique complex, with link phases
  driven by `RateInteraction`s through `codifferential` and `exterior_derivative` of `sin` of the phases, after
  2203.06601 III.
- `gauge_arbitrage_market` in `economics/finance` holds exchange-rate potentials and their rates on the links of a
  lattice complex, evolved by the discrete Maxwell equations with the trader current on links as source, and the good
  on sites moved by a covariant `TransferInteraction` over the links.
- `gauge_network_agents` in `social/opinion_dynamics` holds agent phases on nodes and connections on edges, with
  `TransportedCouplingInteraction`, a thin subclass of `CouplingInteraction` in the package whose function is
  `sin(θ_j - θ_i - A_ij)`, and reads `CellComplexAnalysis(kind='curvature')` and `kind='betti'`.

## Risks / Trade-offs

- [Python loops over links are slow in four dimensions] → The colored schedule updates whole classes vectorised, and
  the gallery uses 2D lattices.
- [Metropolis decorrelates links slower than heat-bath on U(1)] → The scenarios allow the longer runs, and an exact
  heat-bath move can follow without changing the interface.
- [A face sum read as a holonomy holds only for abelian groups] → Non-abelian links are a non-goal, and the docstring of
  `CofaceNeighborhood` says so.
