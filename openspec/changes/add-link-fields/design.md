# Design

## Context

See proposal.md for the motivation. After `redesign-system-model`, a system is its components, interactions, space,
and time. A `Group(on='sites')` holds one member per site of a `LatticeSpace` or node of a `NetworkSpace`, and a group
with `on='edges'` or `on='hyperedges'` holds one member per edge or hyperedge of a `NetworkSpace(hyperedges=)`;
validation reports a group on edges or hyperedges outside a network space. Site groups read neighbours through
`laplacian`, `gradient`, `divergence`, and `advection`, which the compiler lowers to stencils, spectral multipliers,
finite volumes, or finite elements, all on site values. `Sample` is the one Monte Carlo law: its energy, move, and
acceptance parts compile to `CompiledMonteCarlo`, which proposes a new value per site and accepts it from the change of
a local energy over a site neighbourhood, swept once per unit of `DiscreteTime`. `Domain.ANGLE` already exists for
planar spins. Nothing yet reads an edge group with exterior calculus, orients the hyperedges of a network as faces, or
samples a state on edges. The `arbitrage_traders` model of `economics/finance` writes Schwichtenberg's connection as a
site parameter `A_x(t)`, because the framework has no laws on link states.

## Goals / Non-Goals

**Goals:**

- Write a lattice gauge model as its link variables, its plaquette action, and its sampler, in the words of
  Schwichtenberg (2020 §2.1, §4.4) and of Wilson's action, without hand-written index arithmetic.
- Give the same cell states and exterior calculus to non-physics models, such as flows on transport networks and
  exchange rates on a currency graph.
- Keep every existing site-based model, law, and stepper unchanged.
- Add no component kind and no field term to the core: link and plaquette states are edge and hyperedge groups, and the
  physics-named action and observables live in the model package.

**Non-Goals:**

- Fermions, hybrid Monte Carlo with pseudofermions, and SU(3) production runs of lattice QCD. They need sparse Dirac
  solvers and the performance of Grid, and the general capability does not depend on them.
- Simplicial complexes of unstructured meshes. Cubical complexes of lattices and graph complexes with chosen faces
  cover the inventory models and the network uses, and a mesh complex can follow on the same interface.
- Quantum Hamiltonian lattice gauge theory. It belongs with the quantum operators of a separate change.
- States on cells of degree 3 or more. Cubes and larger simplices take part in incidence and topology only.

## Library survey

The constitution asks for a popular, mature, or performant library before any in-house code. The survey checked PyPI
metadata, the repositories, and the licences in October 2026, against the needs of the project: an MIT-compatible
licence or an optional extra, wheels for Linux and macOS on Python 3.12 and 3.13, maintenance, and an interface that
the compiler can lower symbolic laws to.

| Library | What it provides | Licence | Packaging and maintenance | Verdict |
|---|---|---|---|---|
| Grid | Data-parallel C++ lattice QCD library, SU(N) gauge fields, HMC | GPL-2.0 | Built from source with autotools, no Python package, no wheels | Rejected: copyleft, no Python interface, no wheels |
| GPT (Grid Python Toolkit) | Python front end of Grid for lattice QCD | GPL-2.0 | Needs Grid built from its `feature/gpt` branch, or Docker, no wheels | Rejected: copyleft, no wheels, needs a C++ toolchain at install |
| Chroma and QDP++ | C++ lattice QCD application suite | Own licence | C++ build, no Python package | Rejected: no Python interface |
| `lyncs` | Python bindings to lattice QCD libraries | BSD | 0.0.3 from 2020, sdist only, needs the wrapped C libraries | Rejected: unmaintained, no wheels |
| pyQCD | Wilson gauge action and heat-bath in C++ and Cython | Repository licence | Not on PyPI, built from source with Eigen, colours fixed at code generation, SU(N) fundamental only | Rejected: no wheels, no U(1) or Z_N, no graphs |
| `lgt` | Lattice gauge theory in Python | None declared | 0.1.5 from 2022, `requires_python <3.11` | Rejected: no licence, no Python 3.12 or 3.13 |
| `edlgt` | Exact diagonalization of quantum lattice gauge Hamiltonians | Apache-2.0 | 0.2.5 from March 2026, pure Python wheel | Rejected for this change: quantum Hilbert spaces, not classical link fields or Monte Carlo. A candidate for a quantum change |
| `SU2xSU2` | Hybrid Monte Carlo of the SU(2) principal chiral model in 2D | Open source | Pure Python wheel | Rejected: a sigma model on sites, not a gauge field |
| PyDEC (Hirani) | Discrete exterior calculus on simplicial and cubical complexes | BSD-3-Clause | GitHub only, last commit June 2025. The PyPI name `pydec` belongs to an unrelated PyTorch project | Rejected: not installable from PyPI, so it cannot be declared as a dependency. Its cubical coboundaries are a few sparse incidence matrices |
| `fdtd` | Yee finite-difference time-domain electromagnetics | MIT | 0.3.5 from December 2023, pure Python wheel | Rejected: its own grid object model, no symbolic laws, no compact links, no graphs |
| MEEP (`pymeep`) | FDTD electromagnetics | GPL-2.0 or later | conda-forge only | Rejected: copyleft, no PyPI wheels |
| NetworkX | Oriented incidence matrices, cycle bases, greedy coloring | BSD-3-Clause | Already a dependency | Wrapped: network complexes, faces, and the colored schedule of network links |
| SciPy | Sparse matrices, sparse least squares, Bessel functions | BSD-3-Clause | Already a dependency | Wrapped: incidence and Hodge star matrices, the Hodge decomposition, and the exact plaquette values in tests |
| NumPy | `Generator.vonmises`, vectorised gathers | BSD-3-Clause | Already a dependency | Wrapped: the exact U(1) heat-bath draw and the NumPy backend |
| Numba | Compiled loops | BSD-2-Clause | Already a dependency | Wrapped: the compiled backend of holonomies, staples, and SU(2) updates |

No maintained Python library implements compact link variables, Wilson-action Monte Carlo, and exterior calculus on
lattices and graphs with Linux and macOS wheels and a permissive licence. The feature is therefore built in house, as
thin code over NumPy, SciPy, NetworkX, and Numba, which the project already depends on. The in-house parts are the
index tables of cubical complexes, the group arithmetic of U(1), Z_N, and SU(2), the Kennedy-Pendleton and Creutz
heat-bath draws for SU(2), and the overrelaxation reflection, each a few dozen lines of textbook algorithm. No new
runtime dependency and no optional extra are added.

## Decisions

### Cell complexes are network spaces

`CellComplex(space, faces=None, dimension=2)` in `core.space` derives the cells of a space and is itself a network
space, of kind `SpaceKind.NETWORK`, so the groups and bindings of the redesign apply to it without a new space kind. Its
nodes are the sites or nodes of the underlying space, its edges the oriented links or edges, and its hyperedges the
oriented plaquettes or faces, each a sequence of nodes in cyclic order, which fixes its orientation.

For a `LatticeSpace` of `d` dimensions it numbers the links as `(direction, site)` and the plaquettes as
`(plane, site)`, with planes ordered `(0, 1), (0, 2), ...`, keeps the spacings and shape of the lattice, and builds the
coboundary matrices `d0`, `d1`, and `d2` as SciPy CSR matrices with entries `±1`. Fixed and reflecting edges drop the
cells that leave the box, so the complex has a boundary. For a `NetworkSpace` it takes the oriented incidence matrix
from `networkx.incidence_matrix(oriented=True)`, and the faces from the given node cycles, from the network's
hyperedges, or from `networkx.minimum_cycle_basis`. Cells of degree 3, such as the cubes of a 3D lattice, are held for
the incidence matrices and the Betti numbers, without states. The complex also holds, for every cell, the cells of the
next degree that contain it, padded to the largest count, which is the gather table the kernels read. The complex is
derived state, cached on the underlying space by its shape and boundary, so one fact lives in one place.

Declaration: `CellComplex` declares the network space kind with the parts `sites`, `edges`, and `hyperedges`, so
`Group(on='edges')`, `Group(on='hyperedges')`, and the bindings `'neighbours'` and `'hyperedges'` validate on it. A
face whose node cycle uses a missing edge raises a named error when the complex is built, since it is a value error of
the space.

Alternatives considered:
- A separate `LinkSpace` would duplicate the boundary and spacing logic of `LatticeSpace`.
- Letting `LatticeSpace` offer edges and hyperedges directly would change the redesign's rule that groups on edges need
  a network space. The complex keeps that rule; the choice is recorded as an open question.

### Link and plaquette states are edge and hyperedge groups

A state on links is a state of an element placed in `Group(label, element=..., on='edges')` on a cell complex, and a
state on plaquettes one placed in `Group(..., on='hyperedges')`. The redesign's group storage, vectorisation, and
values apply, so every solver, observer, and backend sees an ordinary vector. A potential on links and a field strength
on plaquettes are two groups, or the field strength is the expression `curl(A)` of the link group. Trajectories reshape
an edge group on a lattice complex to `(frames, n_dim, *shape)` and a hyperedge group to `(frames, n_planes, *shape)`,
and to `(frames, n_cells)` on a network complex.

Alternative considered: one site group per direction. It cannot express plaquettes, and the coupling between
directions would need interactions with roles for what is one law.

### Exterior calculus

`exterior_derivative`, `codifferential`, and `hodge_star` are Sage builtin functions in `core.laws`, like the existing
`laplacian`. The degree of their argument is the part of the space its group lives on: 0 for sites, 1 for edges, and
2 for hyperedges. On a state of an edge or hyperedge group, `gradient`, `divergence`, `laplacian`, and the new `curl`
dispatch to them, so the names a user already knows keep their meaning, and on site groups they keep their existing
stencils. A law that couples groups of two degrees, such as a current on links and a charge on sites, is a law of an
interaction whose roles bind the two groups. The compiler lowers the operators to the sparse coboundary matrices on the
NumPy backend and to gather loops over the incidence tables on the Numba backend. The Hodge star of a lattice is
diagonal, the ratio of dual to primal cell volumes from the spacings, and on a network it is the diagonal of declared
edge and face weights, one by default. The codifferential is `(-1)^k ⋆⁻¹ d ⋆`, so `δδ = 0` holds exactly as `dd = 0`
does.

Declaration: each operator declares `Requires(space={SpaceKind.NETWORK})` for states off the sites, and the degree it
accepts. Validation reports a law whose operator gets a state of the wrong degree, such as `curl` of a site state, with
the issue code `law-degree`, the law's path, and a fix that names the operator of that degree.

### Compact group domains

`CompactGroup` is a state domain in `core.components`, of the new kind `DomainKind.COMPACT_GROUP`, built by
`CompactGroup.u1()`, `CompactGroup.zn(n)`, and `CompactGroup.su2()`. These are mathematical groups, so the name passes
the redesign's application-agnostic rule. A U(1) value is one float, the angle. A Z_N value is one float holding an
integer. An SU(2) value is four contiguous floats, a unit quaternion `(a0, a)` for `a0 + i a·σ`. The domain provides
`product`, `inverse`, `trace` normalised to one at the identity, `project`, which wraps angles, reduces integers modulo
N, and renormalises quaternions, and the exact heat-bath draw. Every compiled update calls `project` on the values it
writes, so values never drift off the group.

Declaration: the domain is valid for `Sample`, `Update`, and `Rule` laws. Validation reports a `Rate` or `Noise` law on
a compact group state with the existing code `law-domain`, and suggests a real or angle state for real-time dynamics.

Alternative considered: SU(2) as a 2 × 2 complex matrix. Quaternions take half the storage, multiply with 16 real
products, and avoid a dependency on the complex-states change.

### Holonomies and staples

`holonomy(state, path)` in `core.laws` takes a plaquette, by default every hyperedge of the complex, or a rectangular
loop of given extents in a plane, or a straight line along an axis closed by periodicity. `staples(state)` is the sum
over the plaquettes of a link of the ordered product of its other three links. The compiler lowers both to precomputed
tables of link indices and orientations and to the group product, in NumPy as vectorised gathers and in Numba as loops.
For an abelian group the holonomy angle is the exterior derivative of the link angles, and the lowering reuses it. Both
need a state of an edge group, and validation reports any other with `law-degree`.

### Monte Carlo on links

`Sample` gains support for edge groups on a cell complex: its energy part may be the local energy of a link, written
with `holonomy` and `staples`. `CompiledLinkMonteCarlo` proposes and accepts on links as `CompiledMonteCarlo` does on
sites, and the Monte Carlo stepper sweeps it once per unit of discrete time. The move parts are:
- Metropolis with the rotate move for U(1), the random-value move for Z_N, and a new group move with a width for SU(2),
  which multiplies by a random element near the identity;
- heat-bath, an exact draw from the local Boltzmann weight through the draw of the compact group: `Generator.vonmises`
  around the staple angle with concentration `β |S|` for U(1), a categorical draw over the N values for Z_N, and the
  Kennedy-Pendleton algorithm for SU(2), with Creutz's algorithm below a staple norm where Kennedy-Pendleton rejects too
  often;
- a new overrelax move, the reflection of the link about the staple direction, which keeps the energy and decorrelates
  faster.

Heat-bath and overrelaxation are general Monte Carlo algorithms on compact groups, which the redesign allows in the
core beside Metropolis. The `'colored'` schedule updates in turn the classes of links that share no plaquette: on a
lattice complex with even sizes, links of one direction on sites of one parity, `2 d` classes in all; on a network
complex, the classes of a greedy coloring of the link conflict graph from NetworkX. Every draw comes from the
simulation's `Generator`, as the existing samplers do, so a seed reproduces a run on both backends.

Declarations: the Monte Carlo solver adds `DomainKind.COMPACT_GROUP` to its supported domains and
`ComponentKind.EDGE_GROUP` to its supported components, under `TimeKind.DISCRETE` and `LawKind.SAMPLE`, and the NumPy
and Numba backends declare the same. Validation reports:
- a colored schedule on a lattice complex with an odd periodic size, with the code `sample-schedule`, naming the axis
  and suggesting an even size or the random-sequential schedule;
- a `Sample` on a hyperedge group, with `solver-unsupported`, since no solver samples plaquette states.

### Physics realizations in the model package

The Wilson action and the gauge observables name physics, so they are realizations, not core names, and they live in
`skcomplex.models.physics.waves`, the package of the one model that uses them:
- `wilson_action(beta)` returns the energy `-β Re tr(U S†) / N` of a link with staple sum `S`, which differs from
  `β Σ_p (1 - Re tr U_p / N)` over the link's plaquettes by a term that does not depend on the link;
- `gauge_transform(configuration, g)` maps every link `U(x, μ)` to `g(x) U(x, μ) g(x + μ)⁻¹`;
- `wilson_loop`, `polyakov_loop`, `topological_charge`, and `gauss_residual` are analysis functions over trajectories,
  built on `holonomy` and the exterior calculus. The topological charge of 2D U(1) is the sum of the plaquette angles
  wrapped to `(-π, π]`, divided by `2π`, and generalises to any closed oriented surface of a complex. The Gauss law
  residual is the largest absolute value of `div E - ρ` over the sites;
- the mean plaquette is a standard observable of the model's catalog entry.

`gauge_network_agents` in `social/opinion_dynamics` defines its own alignment energy, as the redesign asks of a package
of another field, and reads the core's `frustration`.

### Real-time link fields

Rate laws on edge groups compile to the ODE form like any group, under `ContinuousTime`, so `RK4`, `ScipyODE`, and
`VelocityVerlet` run them. Lattice Maxwell writes `A_t = -E` and `E_t = δ dA - J` for the noncompact theory, and
`E_t = δ sin(dA) - J` for the compact one, with `A` a real or angle state. Because the codifferential of the electric
rate is `δδ dA - δJ`, the Gauss law holds to roundoff for every stepper, which the scenarios check.

### Hodge decomposition

`hodge_decomposition(flow, complex)` in `core.analysis` solves two sparse least-squares problems with
`scipy.sparse.linalg.lsqr`, one for the site potential and one for the face potential, and returns the gradient part,
the curl part, and the harmonic remainder. The dimension of the harmonic space is the number of independent cycles that
no face bounds, which is 2 on a periodic 2D lattice.

### Visualization

`LinksVisual` in `core.viz`, registered as `'links'`, logs edge groups as `LineStrips2D`, or `Arrows2D` for flows on
networks, colored by value, with a cyclic color map for angles and for the angle of an SU(2) trace. Hyperedge groups of
a 2D lattice complex log as an image of the lattice shape, and faces of a network complex log as colored polygons. A
lattice of three or four dimensions logs the 2D slice that the visual's `plane` and `slice` arguments select.

### Fit with the pipeline and the runtime

- Symbolic layer: edge and hyperedge states, exterior calculus, holonomies, and energies are Sage expressions, and the
  derived energy change of a link move is symbolic, as for site moves.
- Validation: the declarations above let `validate()` report every misuse at once, before compilation.
- Compilation: the compiler picks the Monte Carlo form for link energies and the ODE form for link rates through its
  internal form table.
- Runtime: `run` returns a `Trajectory`. For Monte Carlo the time axis counts sweeps of a Markov chain, the transient is
  the thermalization, and frames are samples. For Hamiltonian dynamics the time axis is physical time. This is the
  convention the Ising model already follows, so no new form is needed.
- Experiments: ensembles over seeds and sweeps over `β` use the existing experiment runner, which the gallery analysis
  uses for the curve of the mean plaquette against `β`.

### The model

`lattice_gauge_fields(size=32, *, dimension=2, group='u1', n=2, beta=2.0, dynamics='monte-carlo', sampler='heat-bath',
charges=None)` lives in `skcomplex.models.physics.waves`, registers under `physics/waves` with the references
`Schwichtenberg 2020` and `Wilson 1974`, and returns a `System` on a cell complex of a periodic lattice with one edge
group. With `dynamics='monte-carlo'`, the system has `DiscreteTime()`, and the link element holds the compact group
state and a `Sample` law with the Wilson action; the defaults run 2200 sweeps with a transient of 200. With
`dynamics='hamiltonian'`, the system has `ContinuousTime()`, the edge group holds the real potential and electric
field, and a site group holds the charge; the defaults run velocity Verlet. The catalog entry declares the mean
plaquette and the Gauss law residual as standard observables. The gallery example keeps the Monte Carlo run within 300
recorded frames by recording every tenth sweep.

## Risks / Trade-offs

- [Python loops over links are slow in four dimensions] → The colored schedule updates whole classes vectorised on
  NumPy and in parallel loops on Numba, and the gallery uses 2D lattices. Production lattice QCD stays a non-goal.
- [The Kennedy-Pendleton draw rejects often at small staple norm] → Creutz's algorithm takes over below a threshold,
  and both are tested against the exact 2D SU(2) plaquette.
- [The colored schedule needs even periodic sizes] → Validation names the axis and offers the random-sequential
  schedule.
- [Reusing `gradient`, `divergence`, and `laplacian` for edge and hyperedge states could surprise a user] → They
  dispatch on the part of the space only, and validation reports a degree mismatch with the law and the state.
- [Floating-point drift moves SU(2) values off the group] → Every write projects back to the group, and a scenario
  checks the norm after 1000 sweeps.

## Simplicial complexes and frustration (added from arXiv 2509.01581)

Mihaylov and Cacciatori model a complex adaptive system as a gauge theory on a simplicial complex: agents at the
vertices, connections on the edges, and curvature on the faces, so complexity is the obstruction, local or topological,
to a globally consistent state. The cell complexes of this change already hold edge and hyperedge groups and multiply
them around loops; the paper needs three general additions.

- **Clique complexes.** `CellComplex.clique_complex(network, dimension)` takes a network's triangles as its faces, and
  its larger cliques as higher simplices, enumerated with NetworkX's `enumerate_all_cliques` up to a chosen dimension.
  The incidence matrices follow from the oriented simplices, so edge and hyperedge groups and exterior calculus work on
  them unchanged.
- **Frustration.** `frustration` in `core.analysis` reports, for a link field of any compact group, the share of
  plaquettes whose holonomy is not the identity and their energy, the term `1 - Re tr(U) / d` summed over them, which
  is zero for a flat connection.
- **Topology.** `betti_numbers` in `core.analysis` computes the Betti numbers from the ranks of the boundary matrices
  over the rationals, with SciPy sparse rank-revealing factorisation, and a nonzero first Betti number is the
  topological frustration of the paper: loops that no gauge transformation can trivialise. The topological charge of a
  U(1) field, in `skcomplex.models.physics.waves`, generalises from the two-dimensional lattice to any closed oriented
  surface of a complex, as the sum of plaquette angles over the surface divided by `2 pi`.
- GUDHI and Ripser compute persistent homology of point clouds and filtrations, a different task; the Betti numbers of
  one fixed complex are a rank computation the project's sparse linear algebra already covers, so no new dependency is
  added.

## Open Questions

- **Edges and hyperedges of a lattice.** The redesign reports a group on edges or hyperedges outside a network space.
  This change keeps that rule by making `CellComplex` a network space derived from a lattice. The alternative is to let
  `LatticeSpace` offer its links and plaquettes as edges and hyperedges directly, which would change the redesign's
  requirement "Groups have a fixed or variable size and live on parts of the space" and the hyperedge scenario of
  "Interactions bind roles with an arity and a binding". Which one the product wants is open.
- **Field terms in the core.** `frustration` stays in `core.analysis` as the general term for holonomies that are not
  the identity, and `CompactGroup` replaces the former `GaugeGroup`. Whether "frustration" passes the reviewed list of
  forbidden field terms, or moves to the model packages with the gauge observables, is open.
