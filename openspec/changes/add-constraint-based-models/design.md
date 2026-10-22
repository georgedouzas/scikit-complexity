# Design

## Context

The parts of the framework that this change builds on, in the vocabulary of `redesign-system-model`:
- A system is `System(label, components=, interactions=, space=, time=)`. Elements declare states and parameters, and
  interactions bind components through roles and return laws.
- Time is `ContinuousTime()`, `DiscreteTime(...)`, or `StaticTime()`. Static time holds only constraint laws, among them
  `Equation`, `Boundary`, `Invariant`, and `Objective`, and its form and solvers are specified by `add-static-models`.
- The compiler derives the form from the laws it lowers. Nested systems whose forms differ co-simulate, and other
  partitions are read as inputs held between synchronisation points.
- `Group(label, element=, on='sites')` on a `LatticeSpace` puts one member of an element on every site.
- `parameter(adaptive=True)` declares a parameter that laws may change, compiled and recorded like a state.
- Every primitive declares `Requires`, every solver and backend declares `Supports`, and `validate()` reports every
  invalid combination as an `Issue` with a code, a path, and a fix before compilation.
- `Call` evaluates a Python or Numba function inside a law, and drivers and live parameter changes reach laws between
  frames.
- `biology/systems_biology` already has Boolean gene networks built from `Update` laws.

Nothing in the core is a decision variable fixed by a linear optimisation at a point in time, and that is what all
seven models need. The `Objective` law kind exists in the vocabulary, but no solver minimises a linear objective under
linear constraints. Raman's static optimisation approach (Eqs. 9.8–9.9) solves the LP at `t_k`, holds the fluxes, and
integrates the exchange ODEs over `Δt`. This is the co-simulation contract with a new kind of partition.

Two more models, elementary flux modes (Raman §9.8) and carbon-13 metabolic flux analysis (§9.7), were recorded as out of
scope because nothing in them evolves in time. One enumerates the generators of a polyhedral cone, and the other fits
fluxes to labelling data. Both are built on the same networks, and both run through `create` and `run` with frames
indexed by what the computation produces: the modes, or the iterations of the fit.

## Goals / Non-Goals

**Goals:**

- Declare a constraint network with the framework's symbols and solve it, in the default install, under the MIT
  licence.
- Couple the solve to ODEs, maps, co-simulation, and site groups without new runtime machinery. A network contributes
  `Equation`, `Bound`, and `Objective` laws, all of the constraint class, so it is valid under every time kind.
- Take the established analyses, model formats, strain-design algorithms, and community methods from COBRApy,
  StrainDesign, and MICOM through an optional extra, as realizations in `skcomplex.models.biology.realizations`, so the
  core holds none of them.
- Keep the core application agnostic. Pools, processes, and switches mean metabolites, reactions, and genes in
  metabolism, and goods, routes, and facilities in a supply chain.
- Declare the requirements and support of every new primitive and solver, so `validate()` reports misuse.
- Enumerate the elementary modes and extreme pathways of a network exactly, and estimate fluxes from labelling data with
  intervals and a goodness-of-fit test, as runs whose frames are modes or fit iterations.

**Non-Goals:**

- Nonlinear constraint-based methods: thermodynamic FBA, enzyme-constrained models with nonlinear kinetics, and
  resource balance analysis. They can follow on the same network once a nonlinear program solver is adopted.
- Flux sampling (`hopsy`, OptGP). It is a separate analysis and can be added to the extra later.
- Gap filling and model reconstruction (CarveMe, ModelSEED). Models are imported, not built.
- Commercial solvers. COBRApy can still use them when the user installs one, but no test depends on them.
- Shipping genome-scale model files in the package. The E. coli core model is the `textbook` model that COBRApy
  bundles, and the other BiGG models are downloaded on request.
- Enumerating the elementary modes of genome-scale networks. Even E. coli core has too many modes for an exact
  double-description enumeration within minutes, and the time limit says so. Elementary flux vectors of bounded
  polyhedra and minimal cut sets from modes are left out too, since StrainDesign already computes minimal cut sets.
- Isotopically non-stationary and dynamic 13C-MFA. They are ODEs of the labelling states and can follow on the same atom
  maps as a rate-law form.
- Correction of measured distributions for natural isotope abundance. Measurements are taken as corrected, as IsoCor
  (GPL-3.0+) does it, and tracer design by optimisation is left to the user.

## Decisions

### Library survey

Checked on 2026-10-19 against PyPI, GitHub, and spike installs on macOS arm64 with Python 3.13:

| Library | Version (date) | Licence | Wheels for Linux and macOS, 3.12–3.13 | Maintenance | Role | Decision |
|---|---|---|---|---|---|---|
| COBRApy (`cobra`) | 0.32.1 (2026-08-11) | LGPL-2.0+ or GPL-2.0+ | pure Python, `py2.py3-none-any`. Its native dependencies `swiglpk` and `python-libsbml` ship cp312 and cp313 wheels | active, last push 2026-10-12, CI on Linux, macOS, and Windows with 3.10 and 3.14 | FBA, FVA, pFBA, MoMA, ROOM, deletions, SBML and JSON I/O | adopt, behind the extra |
| optlang | 1.9.1 (2026-06-02) | Apache-2.0 | pure Python | active (2026-09) | solver layer of COBRApy | transitive only |
| `swiglpk` (GLPK) | 5.0.13 (2026-02-05) | GPL-3.0 | cp312 and cp313 wheels | active | default solver of optlang | transitive only, never imported |
| `highspy` (HiGHS) | 1.15.1 (2026-07-02) | MIT | cp312, cp313, and cp314 for manylinux x86_64 and aarch64, and macOS x86_64 and arm64 | very active, last push 2026-10-19 | LP, MILP, and QP | adopt as a runtime dependency |
| StrainDesign | 1.19.1 (2026-07-26) | Apache-2.0 | pure Python. It needs `pyscipopt` 6.3.0 (MIT, which bundles Apache-2.0 SCIP), with cp313 wheels | active (2026-10) | OptKnock, RobustKnock, OptCouple, minimal cut sets | adopt, behind the extra |
| MICOM | 0.39.1 (2026-08-11) | Apache-2.0 | pure Python. Needs `osqp` 1.1.3 (Apache-2.0), `symengine` 0.14.1 (MIT), and `highspy`, all with wheels | active (2026-10) | community models, cooperative trade-off | adopt, behind the extra |
| `python-libsbml` | 5.21.2 (2026-09-17) | LGPL-2.1 | cp312 and cp313 wheels | active | SBML parsing under COBRApy | transitive only |
| cameo | 0.13.6 (2021-11-09) | Apache-2.0 | pure Python | inactive since 2022, pins `gnomic==1.0.1` | FSEOF, OptGene | reject |
| SteadyComPy | none on PyPI | MIT | none | last push 2017, unfinished port | SteadyCom | reject |
| cometspy | 0.6.3 (2026-02-18) | GPL-3.0 | pure Python, needs the Java COMETS engine | active | COMETS front end | reject |
| `dfba` | 0.1.8 (2020-08-31) | GPL-3.0+ | no wheels, C++ build from source | inactive | dFBA with SUNDIALS and GLPK | reject |
| mergem | 1.1.0 (2024-03-02) | GPL-3.0 | pure Python | low activity | merging models | reject, MICOM covers communities |

The spike on the E. coli core model (`cobra.io.load_model('textbook')`, 72 metabolites and 95 reactions) measured:
- aerobic growth 0.8739215, anaerobic growth 0.2116629, and a pFBA total flux of 518.42209;
- growth 1.9e-15 after knocking out `gapA` (b1779), and 0.86316 after knocking out `pgi` (b4025);
- 7 essential genes in a single-gene deletion scan.

Solving the FBA 200 times with a changing glucose bound took:

| Path | Time per solve |
|---|---|
| `scipy.optimize.linprog(method='highs')`, rebuilt each call | 1.76 ms |
| COBRApy `slim_optimize` | 0.23 ms |
| `highspy` with the model kept and only the column bounds changed | 0.072 ms |

All three gave the same optimum, 0.4178894.

Two solver findings:
- With the `pgi` knock-out against the wild type, ROOM returned zero growth on GLPK. On optlang's `hybrid` interface
  (HiGHS and OSQP), it returned 0.7465 for the linear form and 0.8474 for the MILP form. MoMA gave 0.8426 linear and
  0.8178 quadratic.
- optlang 1.9.1 rejects `'highs'` as an interface name with `highspy` 1.15.1, and only `'hybrid'` reaches HiGHS.

StrainDesign with SCIP found an OptKnock design for succinate with up to three knock-outs and growth of at least 0.1 in
1.9 s, `{PGI, ACKr, CO2t}` and its equivalent sets. Its growth is 0.165 and the smallest succinate secretion at maximal
growth is 9.665, against 0 for the wild type. MICOM built its four-member test community and ran the cooperative
trade-off in 0.4 s.

### Library survey: enumerating the flux cone

Checked on 2026-10-19 against PyPI and GitHub, with spike installs in a fresh Python 3.13 environment on macOS arm64:

| Library | Version (date) | Licence | Wheels for Linux and macOS, 3.12–3.13 | Maintenance | Decision |
|---|---|---|---|---|---|
| `pplpy` (Parma Polyhedra Library) | 0.9.0 (2026-03-24) | GPL-3.0+, with `gmpy2` 2.3.2 and `cysignals` 1.13.1 under LGPL-3.0+ | cp312, cp313, and cp314 for manylinux and musllinux x86_64 and aarch64, and macOS 15 x86_64 and arm64; installed from wheels in the spike | maintained by the SageMath developers, last push 2026-03 | adopt, behind the new extra `polyhedra` |
| `pycddlib` (cddlib) | 3.0.2 (2025-01-13) | GPL-2.0+ | Windows wheels only. The spike's build failed for want of `cddlib/setoper.h` | active, last push 2026-08 | reject, no Linux or macOS wheels |
| `efmtool` | 0.2.1 (2023-03-23) | BSD | pure Python, but drives the Java efmtool through `JPype1`, so it needs a JVM. The spike installed it and failed to start without one | no release since 2023 | reject, a JVM cannot be a pip dependency |
| `efmtool_link` | 0.0.9 (2025-03-12) | Apache-2.0 | sdist only, needs Java and `cobra` | CNApy's link | reject, as `efmtool` |
| efmlrs | 2.3.6 (2021-06-25) | GPL-3.0 | pure Python, pins `cobra==0.17.1`, `numpy==1.18.1`, and `pandas==1.0.1`, and needs the external `mplrs` | inactive | reject, uninstallable on 3.12 |
| ecmtool | 0.1.12 (2021-06-05) | MIT | sdist only, on `pycddlib` or `mplrs` | last push 2024-06 | reject, computes conversion modes, not modes, and inherits `pycddlib`'s builds |
| `polytope` | 0.2.5 (2024-03-07) | BSD | sdist | low activity | reject, no exact ray enumeration |

The spike, run on `pplpy` 0.9.0:
- Raman's toy network of Table 8.2, split into 13 irreversible columns, gave 2 extreme rays in 0.2 ms, which are the 2
  elementary modes after the map back. Normalised to `b1 = 1`, both have `vbio = 1/12` and `v6 = 5/3`. With every bound
  at 1000, the FBA optimum of growth 50 takes `v6 = 1000`, so it is 1000/20, and the FVA ranges of Eq. 8.12 are the
  mixtures of the 2 modes.
- The circulation cone of the complete directed graph on 4 nodes gave 20 extreme rays, the 20 simple cycles of
  `networkx.simple_cycles`.
- E. coli core (95 processes, 46 reversible) with a glucose-only aerobic medium, and an anaerobic variant of 91 processes,
  did not finish within 4 minutes. Genome-scale enumeration is a non-goal, and the time limit turns it into an error.

### Library survey: labelling simulation and flux estimation

| Library | Version (date) | Licence | Install on 3.12–3.13 | Maintenance | Decision |
|---|---|---|---|---|---|
| FreeFlux | 0.3.8 (2026-03-18) | GPL-3.0 | pins `numpy<1.23`, `scipy<1.9`, `pandas~=1.0`, and `pyomo<=6.5`, which have no 3.12 wheels | last push 2026-05 | reject, uninstallable |
| 13CFLUX v3 (`x3cflux`) | 3.1.2 (2026-10-15) | AGPL-3.0 | cp312 and cp313 wheels for Linux x86_64 and macOS arm64 only. It requires `cyipopt` 1.7.0, which is sdist only, and the spike's install failed building it | very active, the fastest engine | reject for now, no aarch64 Linux or Intel macOS wheels and a dependency that needs a system Ipopt |
| mfapy | 0.6.3 (2025-07-07) | MIT | not on PyPI. Installs from GitHub as pure Python with `nlopt` 2.11.0 wheels, and imports on 3.13 | low activity | reference in the test group, not a dependency, since a published extra cannot name a Git URL |
| influx_si | 7.4.3 (2026-02-09) | GPL-2.0+ | sdist, needs R ≥ 3.4 with `nnls`, `rmumps`, and other R packages | active | reject, needs an R runtime |
| INCA, Metran, OpenFLUX | n/a | proprietary or MATLAB | MATLAB | n/a | reject |
| IsoCor | 2.2.4 (2026-06-16) | GPL-3.0+ | pure Python | active | out of scope, natural abundance correction |

The spike ran mfapy's TCA example of Antoniewicz et al. (2007), after renaming its stale `set_constrain` call:
- the simulated glutamate distribution was `[0.3464, 0.2695, 0.2708, 0.0807, 0.0286, 0.0039]`, the published one;
- a fit to noise-free data from two parallel tracers, 50% [1-13C] and 50% [1,2-13C] acetyl-CoA, with 20 starts took
  1.1 s. It recovered every net flux exactly with a residual sum of squares of 8e-32, but put the fumarase exchange at
  72.4 for a true 75, so the data do not identify that exchange.

So no maintained library can be a dependency, and the elementary metabolite unit (EMU) algorithm and the fit are built
in house. Both are thin over SciPy: a sparse LU per unit size and `least_squares`. mfapy is the parity reference in the
tests, and an adapter to `x3cflux` can follow once its wheels cover the supported platforms.

### Licensing: highspy in the core, COBRApy behind an extra

The project is MIT. COBRApy is dual LGPL-2.0+ or GPL-2.0+, and it always installs `swiglpk`, which is GPL-3.0.
- The default install SHALL hold no copyleft package, so COBRApy, StrainDesign, and MICOM go into an optional extra:
  `[project.optional-dependencies] metabolism = ["cobra>=0.32.1", "straindesign>=1.19.1", "micom>=0.39.1"]`.
  StrainDesign and MICOM are Apache-2.0, but both require `cobra`, so they share the extra.
- Only `skcomplex.models.biology.realizations._cobra` imports them, lazily inside its functions. This is the one lazy
  import the constitution allows, with a suppression whose reason names the extra. The core never imports them.
- When the import fails, the adapter raises `MissingExtraError`, a `BackendError` defined in `skcomplex.core.base`.
  Its message names the function, the extra, and the command `pip install 'scikit-complexity[metabolism]'`, and says
  that `ConstraintNetwork` solves, `fseof`, and `steadycom` run without the extra.
- The package uses COBRApy under the LGPL option and never imports `swiglpk` itself. The user guide states that
  installing the extra brings GPL-3.0 code (GLPK) into the environment. A redistributed environment with the extra is
  then governed by that licence, but the MIT source is not.
- deptry learns that the three packages are optional imports.
- `pplpy` is GPL-3.0+ and goes into its own extra, `polyhedra = ["pplpy>=0.9.0"]`, because enumeration is a general
  polyhedral capability that does not need COBRApy. Only `skcomplex.core.analysis._cones` imports it, lazily, and a
  missing extra raises `MissingExtraError` naming `polyhedra`. The user guide states that the extra brings GPL-3.0 PPL
  and LGPL-3.0 `gmpy2` and `cysignals`.

Can optlang or HiGHS alone replace COBRApy for the static LP path?
- optlang alone does not help. It is only a modelling layer, it depends on `swiglpk` (GPL-3.0) and `sympy`, it holds no
  FBA, FVA, or deletion method, and it adds 0.15 ms of overhead per solve.
- `highspy` alone does suffice for the numerics of one solve, an LP, MILP, or convex QP. It is MIT, it has wheels for
  every supported platform, and it is the engine SciPy already uses. It was 25 times faster than `linprog` and 3 times
  faster than COBRApy on repeated solves, and the dynamic and lattice models need exactly that.
- So the core solves a declared network with `highspy`. Lexicographic objectives are a general LP feature that dynamic
  FBA needs for unique fluxes, and they come with that solve.
- Every named analysis that COBRApy implements stays in COBRApy, and is reached by converting the network: FVA, pFBA,
  MoMA, ROOM, deletions, and SBML and JSON I/O. The constitution forbids reimplementing what a mature library provides.
  COBRApy's licence keeps it out of the default install, not out of the project.

Alternatives considered:
- SciPy's `linprog`, which needs no new dependency, was 25 times slower and cannot keep a model between solves. The
  gallery's COMETS example, at about 4·10⁴ solves, would not fit the 60 s budget with it.
- Making COBRApy a runtime dependency would bring GPL-3.0 GLPK into every install.
- Reimplementing FVA, MoMA, and ROOM on `highspy` would break the rule to take capabilities from mature libraries.

### Declaring a network

```python
from skcomplex.core.components import ConstraintNetwork, Constraint, Process
# Symbols, either, both, and total_flux are the core's expression helpers.

A, B, C = Symbols('A B C')
cell = ConstraintNetwork(
    'cell',
    processes=(
        Process('uptake', {A: 1}, lower=0, upper=10),
        Process('convert', {A: -1, B: 1}, rule=either('g1', 'g2')),
        Process('growth', {B: -2, C: 1}, lower=0),
        Process('secrete', {C: -1}, lower=0),
    ),
    switches=('g1', 'g2'),
    constraints=(Constraint(cell_flux_expression, upper=5),),
    objectives=(('growth', 'max'), (total_flux(), 'min')),
)
```

The parts of the declaration:
- Pools are the network's rows. They are symbols or names, and `balance` maps a pool to a non-zero right-hand side,
  `0` by default, which keeps it at steady state.
- A process is a column, with a coefficient per pool and constant bounds, default `0` and `1000`, or `-1000` with
  `reversible=True`. Bounds are written in the units of the user's model.
- Each process exposes a flux symbol, `cell.flux('convert')`. Laws of other elements read it through roles like a
  state, and observables reduce over it.
- A switch is an adaptive parameter, `parameter(adaptive=True)`, with the domain `{0, 1}`, default 1. A rule is a
  Boolean expression in `either`, `both`, and switch names, the same functions laws already use. When its rule is
  false, a process's bounds become 0, which is COBRApy's gene-reaction semantics.
- A knock-out sets a switch to 0. This works as a parameter at construction, as a live parameter change, or as an
  algebraic `Equation` law of an interaction bound to the network, since the redesign lets such laws change adaptive
  parameters. Like every adaptive parameter, a switch is recorded in the trajectory.
- `Constraint(expression, lower=None, upper=None)` adds a row over flux symbols, such as an enzyme capacity or a
  maintenance demand.
- `objectives` is a sequence of `(expression or process name, 'max' | 'min')`. The solve optimises them in order, and
  fixes each optimum within `optimality=1e-9`, relative, before the next. `total_flux()` is the sum of absolute fluxes,
  which the lowering splits into the positive and negative parts of reversible processes.
- `infeasible='raise' | 'zero' | 'hold'` picks what an infeasible solve does: raise `InfeasibleError`, set all fluxes
  and the objective to 0, or keep the last fluxes.
- A network has no `space=` argument. Placing it in `Group(label, element=network, on='sites')` of a system whose space
  is a `LatticeSpace` or a `NetworkSpace` puts one copy on every site.
- Construction validates the declaration against a schema. It rejects an unknown pool or switch, a lower bound above
  the upper bound, a rule over undeclared switches, and an objective that is not linear, naming the process or the
  constraint.

`ConstraintNetwork` is an `Element` of `core.components` built without subclassing, so it stores its arguments
unmodified. Its derived state is `stoichiometry_`, a SciPy sparse matrix, and `processes_`, `pools_`, and `switches_`.
Its fluxes are states of the real domain that only its constraint laws determine. The laws it contributes are:
- one algebraic `Equation` per pool, `S v = b`;
- one `Bound` per process for its declared bounds and rule;
- one `Objective` per entry of `objectives`, with its position in the lexicographic order.

Alternatives considered:
- A table, a matrix and bound vectors, as the only input. It is kept as `ConstraintNetwork.from_arrays(S, lower,
  upper, ...)`, but symbols are the primary form, so bounds and rules can be expressions.
- Reaction strings like COBRApy's. They are metabolism-specific, and import covers files.

### The bound law and switch equations

- `Bound(flux, lower=None, upper=None)` in `core.laws` is a law an interaction returns. Its expressions may read any
  state, parameter, or input. At each solve, the effective bounds are the declared bounds intersected with every
  `Bound` on that flux: the largest lower bound and the smallest upper bound. It is the new law kind `LawKind.BOUND`, of
  the constraint class, so it is valid under every time kind and never changes the form.
- A switch is set from an expression, such as the state of a Boolean node, by an algebraic `Equation` on the adaptive
  switch parameter. A value above 0.5 counts as on. The redesign's adaptive parameters replace the `Switch` law this
  change first planned.
- The `Objective` laws of a network replace the `Solve` law this change first planned. The user never writes them; the
  network contributes them from `objectives`.

The compiler lowers the network to a `ConstraintProblem`:
- the sparse matrix and the constant bounds;
- compiled functions on NumPy or Numba for the variable bounds, the switch values, and the rule mask;
- the split of the objectives.

`HighsLP` in `skcomplex.core.solvers` passes the model to `highspy.Highs` once. Before each solve it changes only the
column bounds with `changeColsBounds`, so HiGHS starts from the last basis. Lexicographic objectives are solved one
after another, each fixing the last optimum as a row. The result holds:
- the fluxes, the objective values, and the status;
- the pool duals (shadow prices) and the reduced costs.

An infeasible solve under `infeasible='raise'` asks HiGHS for an irreducible infeasible subset. `InfeasibleError`
names the network, the time, the site, and the processes and pools in that subset.

### Forms and coupling

The form is derived from the laws the compiler lowers. A network's `Equation`, `Bound`, and `Objective` laws are all of
the constraint class, so they never change the form of the system they are in; its time decides how often it is solved:

| Network in a system with | Time | Form | Meaning |
|---|---|---|---|
| nothing else | `StaticTime()`, inferred when only constraint laws are present | constraint | `ConstraintStepper` solves once per frame with the current parameters and drivers, so a sweep of `add-static-models` makes a phenotype phase plane |
| `Rate` laws | `ContinuousTime()` | ODE | the drift solves first at each evaluation, caching on `(t, y)`, then evaluates the rates that read fluxes. This is the direct approach |
| `Update` laws | `DiscreteTime()` | map | each step solves with the switches of the last step, then applies the updates |
| a sibling nested system | the sibling's time | co-simulation | the constraint partition solves at each synchronisation point, and other partitions read its fluxes as held inputs. This is the static optimisation approach, exact for the exchange ODEs over each interval |

The static optimisation approach is the default in the library's dynamic models, as in Raman's Eqs. 9.8–9.9 and
DFBAlab's practice.
- Its error against the direct approach falls linearly with the synchronisation interval, which the co-simulation
  convergence requirement already covers.
- The direct approach inside an adaptive integrator meets a right-hand side that is only piecewise smooth. The library
  models use `RK4` or `ScipyODE(method='LSODA')` with a maximum step, and the user guide warns about it.

Alternatives considered:
- A `Call` per law. It would solve once per law instead of once per evaluation, and could not share the basis.
- A separate dFBA solver class. That would duplicate co-simulation.

### Per-site networks (COMETS)

A network in a `Group(on='sites')` of a system on a `LatticeSpace` lowers to one `ConstraintProblem` and an array of
bounds per site. Interactions with the site groups of the metabolites bind per site, as other site-group laws do. In COMETS, uptake bounds are Michaelis-Menten in the site's metabolite
concentrations, biomass grows by its growth flux, and metabolites and biomass diffuse with `laplacian`.
- `ConstraintNetwork(active=expression)` skips the solve at sites where the expression is false, such as biomass below
  a threshold, and gives those sites zero fluxes. This is COMETS's own rule, and it keeps the cost proportional to the
  colony, not the lattice.
- Each active site warm-starts from its own last basis, kept per site, and solves run in a loop over active sites.
- The partition solves at synchronisation points, and the partition of the metabolite site groups integrates diffusion
  and exchange with the held fluxes.

Alternative: cometspy drives the Java engine and is GPL-3.0. It cannot run the framework's site groups or solvers, and it
adds a JVM.

### Regulatory FBA

Regulation is a nested system of Boolean nodes with `Update` laws under `DiscreteTime()`, as in the library's gene
networks. An interaction returns `Equation` laws that set the network's adaptive switch parameters from the nodes, and
`Update` laws that set a node's inputs from the external metabolite states and the fluxes, as in Covert et al. (2001).
Raman's Eq. 9.7 alternates the regulatory step and the solve. It is the co-simulation of a map partition, a constraint
partition, and an ODE partition for the medium, with Gauss-Seidel order and the regulatory time step as the
synchronisation interval.

### Static analyses through the extra

The private module `skcomplex.models.biology.realizations._cobra` holds the adapters. Each takes a network or a system
holding one, and a solution or settings, and returns a tidy `pandas` result indexed by process, switch, or design. These
names are public in `skcomplex.models.biology.realizations`, which the `systems_biology` and `ecology` packages import:
- `to_cobra(network)` and `from_cobra(model)`. Pools become metabolites, processes become reactions, switches become
  genes, and rules become GPR strings.
  - Variable bounds are evaluated at a given state, `state=` and `parameters=`, default the initial values.
  - COBRApy's solver is set to `'hybrid'` (HiGHS and OSQP), because GLPK gave wrong ROOM results in the spike.
- `flux_variability`, `parsimonious_fluxes`, `deletions` (single and double, of processes or switches), `moma` (linear
  or quadratic), `room` (linear or MILP), and `synthetic_lethals`. The last is a double-deletion scan that reports the
  pairs with growth below `threshold` whose single deletions are both viable.
- `strain_designs(network, kind='optknock' | 'robustknock' | 'optcouple' | 'mcs', ...)` builds StrainDesign modules
  with SCIP. It maps the knock-outs back to process and switch names.
- `read_network(path or BiGG id)` and `write_network(network, path)` read and write SBML (`.xml`, `.sbml`) and BiGG
  JSON (`.json`) through `cobra.io`. A BiGG identifier is fetched with `cobra.io.load_model`, which caches downloads,
  and `'textbook'` is the bundled E. coli core model.
- `cooperative_tradeoff(networks, abundances, medium, fraction)` builds a MICOM `Community` from the members' COBRApy
  models, written to a temporary directory because MICOM's taxonomy reads files. It returns the community and member
  growth rates and the exchange fluxes.

These live in the library, not in `core.backends`: they are not execution backends of a solver, and they name
metabolic methods (gene deletions, strain design, microbial communities, SBML), which the redesign's
application-agnostic rule keeps out of the core. They declare no `Supports`, since they run no system; each validates
its network with `validate()` before converting it.

A parity test compares the core solve with COBRApy's `optimize` on the toy network and on E. coli core within 1e-7.
It compares the core's `total_flux` objective with `pfba` the same way, so the in-house lexicographic objective cannot
drift from the reference.

### In-house analyses on the core solve

Three functions need no extra. `fseof` and `steadycom` are metabolic methods, so they are realizations in
`skcomplex.models.biology.realizations`; `join_networks` is a general composition of networks, so it lives in
`skcomplex.core.analysis`:
- `fseof(network, target, steps=10)` enforces the target flux at `steps` levels between its value at maximal growth
  and its maximum, solves each level with the objectives of the network, and reports the processes whose absolute flux
  rises monotonically with the target. cameo is the only Python library with FSEOF, and it has been inactive since
  2021 and pins obsolete dependencies.
- `steadycom(networks, medium)` follows Chan et al. (2017). All members grow at one rate `μ`, and abundances sum to 1.
  For a fixed `μ` the problem is an LP in the abundance-scaled fluxes, so a bisection on `μ` finds the largest feasible
  rate. It returns `μ`, the abundances, and the exchange fluxes. No maintained Python implementation exists:
  SteadyComPy stopped in 2017, and MICOM states in its documentation that it does not assume equal growth rates.
- `join_networks(networks, shared, abundances=None)` composes members into one network. Each member's pools and
  processes are prefixed with its label, the shared pools form an extracellular compartment, and the objective is the
  abundance-weighted sum of member growth. This gives joint and community FBA (Raman Eq. 11.3) as a plain network, so
  every analysis applies to it.

### Enumerating the flux cone

`skcomplex.core.analysis._cones` holds one general function, `extreme_rays(equalities, nonnegative)`. It takes an
integer matrix of equalities and the indices of the non-negative columns, builds a `ppl.C_Polyhedron`, and returns the
rays and lines of its minimised generators as integer arrays. It knows nothing of networks, so a cone of any field uses
it. It is private to the analysis package rather than a module of `core.backends`, since it runs no system and the
redesign keeps `core.backends` for execution backends.

`skcomplex.core.analysis` builds on it:
- `elementary_modes(network, normalise=None, timeout=60.0)` reads the directions from the bounds: a lower bound below 0
  makes a process reversible, an upper bound of at most 0 reverses its column, and a process whose rule is false or
  whose bounds are both 0 is removed. Finite bounds are ignored, since modes belong to the cone. Each reversible column
  is split in two, the extreme rays of the split cone are taken, the two-cycles of split pairs are dropped, and the rest
  are mapped back. These are exactly the elementary modes (Gagneur and Klamt 2004). The result is a `pandas` table
  indexed by mode and process, scaled to `normalise` or to the smallest integer vector.
- `extreme_pathways(network, exchanges=None, timeout=60.0)` splits only the reversible internal processes and keeps
  each exchange as one column in its declared direction, following Schilling et al. (2000). A process that touches one
  pool is an exchange unless `exchanges` names them. It returns the pathways and, for a cone that is not pointed, the
  lines.
- Coefficients are converted to fractions with a denominator of at most 10⁶, and each row is scaled to integers. A
  coefficient that the conversion moves by more than 1e-9 raises a `ValidationError` that names the process and pool.
- A pool with a non-zero `balance` raises a `ValidationError`, since modes are defined at steady state.
- `pplpy` runs under `cysignals`, so the time limit is a `cysignals` alarm. When it fires, `EnumerationLimitError`, a
  `SolverError`, names the network, its number of processes, and the limit, and suggests closing exchanges or
  enumerating a subnetwork.

`FluxModes(network, kind='modes' | 'pathways', normalise=None, order=None)` is an element of `core.components` that
contributes an `Enumerate` law, of the new law kind `LawKind.ENUMERATE` in the constraint class. `order` is a ratio, such as `('vbio', 'b1')` for the growth yield, that sorts the modes in decreasing
order.

### Atom maps and labelling simulation

A process declares its atom transitions with `AtomMap`:

```python
Process('v1', {AcCoA: -1, OAC: -1, Cit: 1}, atoms=AtomMap([(AcCoA, 'ab'), (OAC, 'cdef')], [(Cit, 'fedbac')]))
```

- Each occurrence pairs a pool with letters, so a pool that appears twice, as in a condensation, has two occurrences.
  The letters of the products are a permutation of the letters of the reactants, except for atoms that leave to a pool
  that is not traced.
- A pool's atom count is the length of its letters, and it must be the same in every map. `symmetric=(Suc, Fum)` on the
  experiment marks pools whose atoms are read in either order with equal weight.
- `Tracer(pool, {'10': 0.5, '00': 0.5})` gives the isotopomer fractions of a substrate pool as bit strings, and
  `Fragment(pool, atoms=(1, 2, 3, 4, 5), name='Glu')` names a measured fragment.

`skcomplex.core.compile` lowers an experiment to a `LabellingProblem`:
- From the measured fragments, it walks the atom maps backward and collects the elementary metabolite units (Antoniewicz
  et al. 2007) that they need, grouped by size.
- For each size `n`, it builds sparse matrices with the flux-dependent entries `A_n(v) X_n = B_n(v) Y_n`, where `Y_n`
  holds the convolutions of smaller units and the tracer inputs. The matrix entries are compiled as functions of the
  forward and backward fluxes, on NumPy, like the other compiled laws.
- `EmuSolver` in `skcomplex.core.solvers` solves the sizes in increasing order with `scipy.sparse.linalg.splu`, and
  returns each fragment's mass isotopomer distribution. A singular `A_n` means a pool without flux, and raises a named
  error that names the pool.

`simulate_labelling(experiment, fluxes)` in `skcomplex.core.analysis` returns the distributions for given fluxes. It is
the forward model of the fit and the tool for choosing tracers by hand.

### Flux estimation

`LabellingExperiment(network, tracers, measurements, fluxes=None, symmetric=(), starts=5, intervals='linear')` is an
element of `core.components` that contributes an `Estimate` law, of the new law kind `LawKind.ESTIMATE` in the
constraint class. Several experiments with different tracers share one network and one flux
vector, and are passed as a sequence.
- Measurements map each fragment to a measured distribution and its standard deviations. `fluxes` maps measured or
  fixed fluxes to a value and a standard deviation, or to a value alone for a fixed flux.
- The fluxes are parametrised by a basis of the null space of the network, taken from SciPy's `null_space` and
  reduced to free net fluxes, plus one exchange flux per reversible process. So every balance holds exactly, and the
  forward and backward fluxes are `net⁺ + exchange` and `net⁻ + exchange`.
- `estimate_fluxes(experiment, random_state=None)` minimises the weighted residuals with `scipy.optimize.least_squares`
  (`trf`, bounds from the processes). The starts are feasible points drawn by solving the network with `HighsLP` for
  seeded random linear objectives and averaging pairs of the solutions.
- Linear intervals come from the covariance `(Jᵀ W J)⁻¹` at the optimum. Profile intervals fix one flux on a grid,
  refit, and keep the values whose residual rises by less than the 95% chi-square quantile of one degree of freedom, as
  in Antoniewicz et al. (2006). A profile that reaches a bound marks the flux as not identifiable.
- The goodness-of-fit test compares the minimal residual with the 95% range of the chi-square distribution whose
  degrees of freedom are the measurements minus the free fluxes.
- The result, a `FluxEstimate`, holds the estimates, intervals, residual, test, and simulated distributions.

A test compares `simulate_labelling` with mfapy on the TCA example and on mfapy's E. coli example within 1e-10, and the
published glutamate distribution within 5e-5.

`calibrate` in `skcomplex.core.experiments` fits parameters to time series by running a model. The estimation is a
different problem, a steady-state forward model with a flux parametrisation, so it does not reuse `calibrate`, but it
uses the same `least_squares` settings.

### Runs that are not integrated in time

An enumeration or an estimation is computed, not integrated. Both are systems under `StaticTime()`, and both fit the
`System`, `Simulation`, and `Trajectory` contract by indexing frames by what the computation produces:
- `Enumerate` gives the enumeration form. `ModeStepper` enumerates at the first step and then emits one mode per step,
  in the declared order. The trajectory's `times` are the mode indices 0, 1, …, its axis is `mode`, and the run ends
  after the last mode, whatever `until` asks.
- `Estimate` gives the estimation form. `EstimationStepper` runs `estimate_fluxes` at the first step with the
  `least_squares` callback, which SciPy 1.16 added, and buffers each accepted iterate with its start. It then emits one
  frame per iterate, and a last frame with the best estimate, its intervals, and its test. The axis is `iteration`.
- Neither law can share a system or a partition with laws of another kind, because a frame of modes or iterations has
  no time to synchronise with. `validate()` reports such a system with the issue `constraint.enumeration-alone` or
  `constraint.estimation-alone`, whose path is the element and whose fix is to run it in a system of its own.
- `Trajectory` gains an `axis` attribute, `'time'` by default, shared with `add-static-models`, which adds the same
  attribute for sweeps; whichever change lands first adds it. Observers read it, and the analysis functions return the
  same values as the frames, since the steppers call them.

### Declarations and validation

Every new primitive and solver declares its requirements or support, so `validate()` reports misuse before compilation
and the generated compatibility table lists them:

| Primitive or solver | Declaration | Issue when misused |
|---|---|---|
| `ConstraintNetwork` | `Requires(time={STATIC, CONTINUOUS, DISCRETE}, space={NONE, LATTICE, NETWORK}, domains={REAL})` | a network in a continuous space or on edges, with the fix to place it in a site group |
| `Bound` | `Requires(domains={REAL})`, law class constraint, valid under every time | a bound on a flux that no network of the system declares, naming the flux |
| `Enumerate`, `Estimate` | `Requires(time={STATIC})` | the law under continuous or discrete time, or beside laws of another kind |
| `HighsLP` | `Supports(time={STATIC, CONTINUOUS, DISCRETE}, laws={EQUATION, BOUND, OBJECTIVE}, domains={REAL})` | an objective that is not linear in the fluxes |
| `ConstraintStepper` | `Supports(time={STATIC}, space={NONE, LATTICE, NETWORK}, laws={EQUATION, BOUND, OBJECTIVE}, components={ELEMENT, SITE_GROUP})` | a static system with laws it does not run |
| `ModeStepper` | `Supports(time={STATIC}, laws={ENUMERATE})` | the `polyhedra` extra not importable, reported by the availability check with the install command |
| `EmuSolver`, `EstimationStepper` | `Supports(time={STATIC}, laws={ESTIMATE})` | a pool without flux, naming the pool |

Adaptive switch parameters need no declaration of their own: they are adaptive parameters of the redesign, and an
`Equation` on one is validated like any law on an adaptive parameter.

### Results, observables, and trajectories

Fluxes, objective values, and switch values are recorded per frame under dotted names:
- `cell.flux.convert`, `cell.objective.0`, and `cell.switch.g1`;
- per site, as arrays shaped like the lattice.

Pool duals are recorded on request with `record_duals=True`, so dense duals do not bloat long recordings. Observables,
which `Simulation(observables=...)` and the models' catalog entries take, reduce over fluxes like over states.

An enumeration records `cell.flux.<process>` per mode and `cell.mode.ratio` for the ordering ratio. An estimation
records `tca.flux.<process>`, `tca.exchange.<process>`, `tca.rss`, `tca.start`, and `tca.mdv.<fragment>` per iteration,
and at the last frame `tca.lower.<process>`, `tca.upper.<process>`, `tca.chi2`, and `tca.accepted`.

### Visuals

`FluxVisual` is registered as `'fluxes'` and is the default visual of a non-spatial network. It logs:
- a `rerun.BarChart` of the fluxes in process order under `<path>/fluxes`, with an optional `processes=` subset for
  genome-scale models;
- the objective values as scalars under `<path>/objective`;
- with `graph=True`, the default for at most 100 processes, a `rerun.GraphNodes` of pools and processes and a
  `rerun.GraphEdges` from the coefficients, with edge widths proportional to `|flux|` and colours by sign. The node
  layout is computed once with `networkx.spring_layout`, seeded, so captures are reproducible.

A network on a lattice defaults to `ImageVisual` of a chosen flux per site, `growth` by default, beside the images of the
site groups. Rerun 0.39 has all three archetypes, which the spike checked.

A run whose axis is not time sets the `step` timeline and a sequence timeline named by its axis, `mode` or
`iteration`, instead of `sim_time`, and the blueprint's time panel follows that timeline. `FluxModes` uses `FluxVisual`
unchanged, so each mode is a bar chart and a graph. `LabellingVisual`, registered as `'labelling'`, logs the flux graph
with the current estimate, the residual sum of squares under `<experiment>/rss`, and, per fragment, the measured and
simulated distributions as two `rerun.BarChart`s under `<experiment>/mdv/<fragment>/`.

### Library models and gallery examples

Every model has an example at `docs/examples/biology/<subfield>/plot_<name>.py`, created through the gallery change's
requirements:
- the Rerun primary view is the thumbnail;
- at least one Matplotlib analysis plot comes from the run;
- the budget is at most 300 frames, 20 MB, and 60 s.

Models that need the extra name it in their docstring, and the docs and tests sessions install it.

| Model | Network and setting | Rerun primary view | Analysis plot | Agreement shown |
|---|---|---|---|---|
| `flux_balance` | Raman's toy network, Table 8.2, by default. `network='e_coli_core'` with the extra, with the oxygen uptake bound ramped from 0 to 20 over the frames | flux bar chart | FVA ranges as bars with the pFBA fluxes as points, and growth against oxygen uptake | FBA optimum on the FVA range, growth 0.874 at full oxygen and 0.212 without |
| `metabolic_perturbations` | E. coli core with the switches knocked out one per frame | pool-process graph with fluxes | growth after each single deletion, and FBA, MoMA, and ROOM growth for `pgi` | 7 essential genes at zero growth, and MoMA and ROOM below FBA |
| `regulatory_fba` | E. coli core with Covert-style rules on glucose and acetate, batch culture | biomass, glucose, and acetate series | uptake fluxes against time, with the regulatory switch states | acetate uptake 0 until glucose is depleted, then positive |
| `dynamic_fba` | E. coli core batch, Michaelis-Menten glucose uptake, static optimisation approach | biomass and glucose series | `ln X` against time with a fitted slope, and glucose against biomass | slope 0.874, final biomass `X0 + 0.0874·S0` within 2% |
| `dynamic_multispecies_fba` | two E. coli core strains that differ in uptake `vmax`, one medium | biomass series per strain | strain fractions against time | the faster strain's share rises, and the mass balance closes |
| `comets` | 25×25 lattice, E. coli core, central inoculum, diffusing glucose | biomass field image | colony radius against time, and glucose consumed against the integral of uptake times biomass | linear front speed, and the glucose balance within 1% |
| `community_fba` | two E. coli core members with complementary knock-outs, joint FBA, SteadyCom, and cooperative trade-off | member growth bar chart | community growth against the trade-off fraction, and SteadyCom abundances | community growth at fraction 1 equals the joint FBA optimum |
| `elementary_flux_modes` | Raman's toy network, its 2 modes ordered by growth yield on `A`, one frame per mode | pool-process graph of the mode | the modes as a process-by-mode heat map, and the FBA optimum as stacked contributions of the modes | both yields are 1/12, and the decomposition reproduces growth 50 |
| `carbon_13_mfa` | the TCA example of Antoniewicz et al. (2007), two parallel tracers, data from known fluxes with noise of standard deviation 0.002 at seed 0, 5 starts | pool-process graph of the estimated fluxes | estimated net fluxes with 95% intervals against the true ones, measured against simulated glutamate distributions, and the residual against iteration | every interval contains its true flux, and the fit passes the chi-square test |

`comets` stays within budget by solving only active sites. At about 100 active sites, 200 frames, and two
lexicographic solves of 0.07 ms each, the solves take about 3 s.

## Risks / Trade-offs

- [COBRApy 0.32.1 caps `pandas<3.0`, while the lock holds pandas 3.0.6 through Mesa] → Mesa 3.5 accepts pandas 2.3, so
  the lock resolves pandas 2.3.x for every group. The first task checks the full test suite on pandas 2.3, and records
  the cap in the dependency notes until COBRApy lifts it.
- [GPL-3.0 GLPK comes with the extra] → The core never imports it. COBRApy, StrainDesign, and MICOM are pointed to
  HiGHS (`hybrid`) and SCIP, and the user guide names the licence of the extra's environment.
- [Non-unique optimal fluxes make dynamic trajectories depend on the solver] → The library's dynamic models use a
  lexicographic second objective, the minimal total flux, which makes the fluxes unique in practice. The user guide
  explains it.
- [A depleted substrate makes the LP infeasible, through the maintenance demand] → The dynamic models set
  `infeasible='zero'`, and growth stops as in Raman §9.6. The default stays `'raise'`, so a static model never hides
  infeasibility.
- [The direct approach with adaptive integrators chatters at bound switches] → The static optimisation approach is the
  library default, the direct approach is documented with fixed or capped steps, and both are tested to agree.
- [Downloading BiGG models in tests is flaky] → Tests use the bundled `textbook` model. One test that downloads iNJ661
  is marked `network` and stays out of the default run.
- [The per-site solve dominates COMETS cost] → Active sites only, warm starts per site, and a lattice size chosen for
  the budget. Parallel solves across sites are a follow-up if needed.
- [optlang's `highs` interface does not load with `highspy` 1.15.1] → The adapter uses `hybrid`, which works. A test
  pins the interface name, so an optlang update that changes it fails loudly.
- [The number of elementary modes grows combinatorially] → A time limit with a named error, a documented non-goal for
  genome-scale networks, and subnetworks in the examples.
- [`pplpy` wheels need macOS 15 or later] → The user guide states it, and older macOS builds from source with PPL from
  Homebrew. The first task checks the macOS version of the CI runners.
- [The EMU code and the fit are in house] → Parity tests against mfapy and the published distribution, and the core
  keeps only the EMU lowering and a thin fit over SciPy. The design records the libraries surveyed and why each was
  rejected, as the constitution asks.
- [Local minima in the fit] → Several seeded feasible starts, the best kept, and the residual compared with the
  chi-square range, so a poor local minimum shows as a rejected fit.
- [mfapy is installed from Git in the test group] → It is pinned to a commit and used only by the parity tests, never
  by the package.

## Migration Plan

The change only adds things:
- `highspy` joins the runtime dependencies, and the Project Profile's dependency list grows by it in the same PR;
- the `metabolism` and `polyhedra` extras are added, and the SciPy floor rises to 1.16;
- the nox `tests`, `docs`, and `gallery` sessions install the extra;
- one CI job runs a default install without the extra and checks the missing-extra errors.

The seven inventory entries move from blocked to pending this change, and to implemented as their models land. The
entries `elementary_flux_modes` and `carbon_13_mfa` move from out of scope to pending this change, and to implemented.

## Open Questions

- Whether to add an `x3cflux` adapter for large labelling networks once its wheels cover Linux aarch64 and Intel macOS
  and `cyipopt` ships wheels. The in-house solver stays the default either way.
- Whether the labelling primitives stay in the core. `AtomMap`, `Tracer`, `Fragment`, `LabellingExperiment`, the
  elementary metabolite unit lowering, and `EmuSolver` are named after chemistry and metabolism (atoms, isotopes,
  metabolite units), which the redesign's field-term test may forbid. The two settlements change the scope differently:
  rename them to general terms (a label map over the units of a pool, a unit decomposition) and keep them in
  `core.components`, `core.compile`, and `core.solvers`; or move the whole of `isotope-labelling` to
  `skcomplex.models.biology.realizations`, with its own stepper registered from the library. This design keeps the
  current names and core placement until that decision is made.
- Which change defines the `Objective` law's lowering under static time. The redesign declares the law kind and leaves
  static solvers to `add-static-models`; this change lowers linear objectives with lexicographic order to `HighsLP`.
  If `add-static-models` lands first with its own objective lowering, the two must agree on how a solver claims an
  objective, which the declarations alone do not decide.
- Resolved: the toy network's optimum is in Raman section 8.6: with the bounds of the example (internal fluxes in [0, 1000],
  reversible ones in [-1000, 1000]), growth is 50 (Eq. 8.9), and FVA gives the ranges of Eq. 8.12.
