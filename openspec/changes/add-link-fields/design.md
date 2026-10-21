# Design

## Context

See proposal.md for the motivation. Every group of the core places one member per site of a `LatticeSpace` or per
node of a `NetworkSpace`. A `Field` reads neighbours through `laplacian`, `gradient`, `divergence`, and `advection`,
which the compiler lowers to stencils, spectral multipliers, finite volumes, or finite elements, all on site values.
Monte Carlo laws, `Energy`, `Move`, and `Sampler`, compile to `CompiledMonteCarlo`, which proposes a new value per site
and accepts it from the change of a local energy over a site neighbourhood, swept by the `MonteCarlo` stepper once per
unit of time. `Domain.ANGLE` already exists for planar spins. The `arbitrage_traders` model of `economics/finance`
writes Schwichtenberg's connection as a site parameter `A_x(t)`, because the framework has no state on links.

## Goals / Non-Goals

**Goals:**

- Write a lattice gauge model as its link variables, its plaquette action, and its sampler, in the words of
  Schwichtenberg (2020 §2.1, §4.4) and of Wilson's action, without hand-written index arithmetic.
- Give the same cell states and exterior calculus to non-physics models, such as flows on transport networks and
  exchange rates on a currency graph.
- Keep every existing site-based model, law, and stepper unchanged.

**Non-Goals:**

- Fermions, hybrid Monte Carlo with pseudofermions, and SU(3) production runs of lattice QCD. They need sparse Dirac
  solvers and the performance of Grid, and the general capability does not depend on them.
- Simplicial complexes of unstructured meshes. Cubical complexes of lattices and graph complexes with chosen faces
  cover the inventory models and the network uses, and a mesh complex can follow on the same interface.
- Quantum Hamiltonian lattice gauge theory. It belongs with the quantum operators of a separate change.

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

### Cell complexes

`CellComplex(space, faces=None)` derives the cells of a space. For a `LatticeSpace` of `d` dimensions it numbers the
links as `(direction, site)` and the plaquettes as `(plane, site)`, with planes ordered `(0, 1), (0, 2), ...`, and
builds the coboundary matrices `d0`, `d1`, and `d2` as SciPy CSR matrices with entries `±1`. Fixed and reflecting
edges drop the cells that leave the box, so the complex has a boundary. For a `NetworkSpace` it takes the oriented
incidence matrix from `networkx.incidence_matrix(oriented=True)`, and the faces from the given node cycles or from
`networkx.minimum_cycle_basis`. The complex also holds, for every cell, the cells of the next degree that contain it,
padded to the largest count, which is the gather table the kernels read. The complex is derived state, cached on the
space by its shape and boundary, so one fact lives in one place.

Alternative considered: a separate `LinkSpace`. It would duplicate the boundary and spacing logic of `LatticeSpace`, so
the complex is a view of the existing spaces instead.

### Cell states and storage

`state(..., degree=k)` declares a state on cells of degree `k`, and `degree=0` is the default, so every existing
element is unchanged. `LinkField(label, element, space, values=..., faces=None)` is a `Field` whose states each hold one
value per cell of their degree. The flat state vector stores each state as a contiguous block in the order of the
complex, so every solver, observer, and backend sees an ordinary vector. Trajectories reshape the block to
`(frames, n_dim, *shape)` for links and `(frames, n_planes, *shape)` for plaquettes on a lattice, and to
`(frames, n_cells)` on a network.

Alternative considered: one site group per direction. It cannot express plaquettes, and the coupling between
directions would need interactions with roles for what is one law.

### Exterior calculus

`exterior_derivative`, `codifferential`, and `hodge_star` are Sage builtin functions like the existing `_Laplacian`.
On a state with a degree, `gradient`, `divergence`, `laplacian`, and the new `curl` dispatch to them, so the names a
user already knows keep their meaning, and on site states they keep their existing stencils. The compiler lowers the
operators to the sparse coboundary matrices on the NumPy backend and to gather loops over the incidence tables on the
Numba backend. The Hodge star of a lattice is diagonal, the ratio of dual to primal cell volumes from the spacings, and
on a network it is the diagonal of declared edge and face weights, one by default. The codifferential is
`(-1)^k ⋆⁻¹ d ⋆`, so `δδ = 0` holds exactly as `dd = 0` does.

### Gauge groups

`GaugeGroup` is a domain in `skcomplex.core.base`, built by `GaugeGroup.u1()`, `GaugeGroup.zn(n)`, and
`GaugeGroup.su2()`. A U(1) link is one float, the angle. A Z_N link is one float holding an integer. An SU(2) link is
four contiguous floats, a unit quaternion `(a0, a)` for `a0 + i a·σ`, so the storage rule of the complex states change
is not needed. The group provides `product`, `inverse`, `trace` normalised to one at the identity, `project`, which
wraps angles, reduces integers modulo N, and renormalises quaternions, and the heat-bath draw. Every compiled update
calls `project` on the values it writes, so values never drift off the group.

Alternative considered: SU(2) as a 2 × 2 complex matrix. Quaternions take half the storage, multiply with 16 real
products, and avoid a dependency on the complex-states change.

### Holonomies and staples

`holonomy(state, path)` takes a plaquette, by default every plaquette of the complex, or a rectangular loop of given
extents in a plane, or a straight line along an axis closed by periodicity. `staples(state)` is the sum over the
plaquettes of a link of the ordered product of its other three links. The compiler lowers both to precomputed tables of
link indices and orientations and to the group product, in NumPy as vectorised gathers and in Numba as loops. For an
abelian group the holonomy angle is the exterior derivative of the link angles, and the lowering reuses it.

### Monte Carlo on links

`Energy(expression, cells='links')` declares a local energy of a link, written with `holonomy` and `staples`, and
`WilsonAction(beta)` returns the energy `-β Re tr(U S†) / N` of a link with staple sum `S`, which differs from
`β Σ_p (1 - Re tr U_p / N)` over the link's plaquettes by a term that does not depend on the link. `CompiledLinkMonteCarlo`
proposes and accepts on links as `CompiledMonteCarlo` does on sites, and the `MonteCarlo` stepper sweeps it once per
unit of time. The moves are:
- Metropolis with `Move.rotate` for U(1), `Move.random_value` for Z_N, and `Move.group(width)` for SU(2), which
  multiplies by a random element near the identity;
- heat-bath, an exact draw from the local Boltzmann weight: `Generator.vonmises` around the staple angle with
  concentration `β |S|` for U(1), a categorical draw over the N values for Z_N, and the Kennedy-Pendleton algorithm for
  SU(2), with Creutz's algorithm below a staple norm where Kennedy-Pendleton rejects too often;
- `Move.overrelax()`, the reflection of the link about the staple direction, which keeps the action and decorrelates
  faster.
The `'colored'` schedule updates in turn the classes of links that share no plaquette: on a lattice with even sizes,
links of one direction on sites of one parity, `2 d` classes in all; on a network, the classes of a greedy coloring of
the link conflict graph from NetworkX. Every draw comes from the simulation's `Generator`, as the existing samplers do,
so a seed reproduces a run on both backends.

### Gauge transformations and observables

`gauge_transform(configuration, g)` maps every link `U(x, μ)` to `g(x) U(x, μ) g(x + μ)⁻¹`. `wilson_loop`,
`polyakov_loop`, `topological_charge`, and `gauss_residual` are analysis functions over trajectories, and the mean
plaquette is also an `Observable` that the model records each frame. The topological charge of 2D U(1) is the sum of
the plaquette angles wrapped to `(-π, π]`, divided by `2π`. The Gauss law residual is the largest absolute value of
`div E - ρ` over the sites.

### Real-time link fields

Rate laws on cell states compile to the ODE form like any field, so `RK4`, `ScipyODE`, and `VelocityVerlet` run them.
Lattice Maxwell writes `A_t = -E` and `E_t = δ dA - J` for the noncompact theory, and `E_t = δ sin(dA) - J` for the
compact one. Because the codifferential of the electric rate is `δδ dA - δJ`, the Gauss law holds to roundoff for every
stepper, which the scenarios check.

### Hodge decomposition

`hodge_decomposition(flow, complex)` solves two sparse least-squares problems with `scipy.sparse.linalg.lsqr`, one for
the site potential and one for the face potential, and returns the gradient part, the curl part, and the harmonic
remainder. The dimension of the harmonic space is the number of independent cycles that no face bounds, which is 2 on a
periodic 2D lattice.

### Visualization

`LinksVisual`, registered as `'links'`, logs links as `LineStrips2D`, or `Arrows2D` for flows on networks, colored by
value, with a cyclic color map for angles and for the angle of an SU(2) trace. Plaquettes of a 2D lattice log as an
image of the lattice shape, and faces of a network log as colored polygons. A lattice of three or four dimensions logs
the 2D slice that the visual's `plane` and `slice` arguments select. The default blueprint of a link field shows the
plaquette image and the mean plaquette series.

### Fit with the pipeline and the runtime

- Symbolic layer: cell states, exterior calculus, holonomies, and energies are Sage expressions, and the derived energy
  change of a link move is symbolic, as for site moves.
- Compilation: the NumPy and Numba printers gain the operators above, and `compile_system` picks the Monte Carlo form
  for link energies and the ODE form for link rates through the existing form resolution.
- Runtime: `run` returns a `Trajectory`. For Monte Carlo the time axis counts sweeps of a Markov chain, the transient is
  the thermalization, and frames are samples. For Hamiltonian dynamics the time axis is physical time. This is the
  convention the Ising model already follows, so no new form is needed.
- Experiments: ensembles over seeds and sweeps over `β` use the existing experiment runner, which the gallery analysis
  uses for the curve of the mean plaquette against `β`.

### The model

`lattice_gauge_fields(size=32, *, dimension=2, group='u1', n=2, beta=2.0, dynamics='monte-carlo', sampler='heat-bath',
charges=None)` lives in `skcomplex.models.physics.waves`, registers under `physics/waves` with the references
`Schwichtenberg 2020` and `Wilson 1974`, and returns a `System` with one `LinkField`. With `dynamics='monte-carlo'`, the
link element holds the group-valued state and the Wilson action, and the defaults run 2200 sweeps with a transient of
200. With `dynamics='hamiltonian'`, the element holds the real potential and electric field on links and the charge on
sites, and the defaults run velocity Verlet. Both record the mean plaquette and the Gauss law residual as observables.
The gallery example keeps the Monte Carlo run within 300 recorded frames by recording every tenth sweep.

## Risks / Trade-offs

- [Python loops over links are slow in four dimensions] → The colored schedule updates whole classes vectorised on
  NumPy and in parallel loops on Numba, and the gallery uses 2D lattices. Production lattice QCD stays a non-goal.
- [The Kennedy-Pendleton draw rejects often at small staple norm] → Creutz's algorithm takes over below a threshold,
  and both are tested against the exact 2D SU(2) plaquette.
- [The colored schedule needs even periodic sizes] → Compiling names the axis and offers the random-sequential
  schedule.
- [Reusing `gradient`, `divergence`, and `laplacian` for cell states could surprise a user] → They dispatch on the
  declared degree only, and a degree mismatch raises a named error that names the law and the state.
- [Floating-point drift moves SU(2) values off the group] → Every write projects back to the group, and a scenario
  checks the norm after 1000 sweeps.

## Simplicial complexes and frustration (added from arXiv 2509.01581)

Mihaylov and Cacciatori model a complex adaptive system as a gauge theory on a simplicial complex: agents at the
vertices, connections on the edges, and curvature on the faces, so complexity is the obstruction, local or topological,
to a globally consistent state. The cell complexes of this change already hold states on links and faces and multiply
them around loops; the paper needs three general additions.

- **Clique complexes.** A network's faces are its triangles, and higher simplices are its larger cliques, enumerated with
  NetworkX's `enumerate_all_cliques` up to a chosen dimension. The incidence matrices follow from the oriented simplices,
  so cell states and exterior calculus work on them unchanged.
- **Frustration.** A plaquette is frustrated when its holonomy is not the identity. The measure reports the share of
  frustrated plaquettes and their energy, the Wilson action term `1 - Re tr(U) / d` summed over them, which is zero for a
  flat connection.
- **Topology.** The Betti numbers come from the ranks of the boundary matrices over the rationals, with SciPy sparse
  rank-revealing factorisation, and a nonzero first Betti number is the topological frustration of the paper: loops
  that no gauge transformation can trivialise. The topological charge of a U(1) field generalises from the
  two-dimensional lattice to any closed oriented surface of a complex, as the sum of plaquette angles over the surface
  divided by `2 pi`.
- GUDHI and Ripser compute persistent homology of point clouds and filtrations, a different task; the Betti numbers of
  one fixed complex are a rank computation the project's sparse linear algebra already covers, so no new dependency is
  added.
