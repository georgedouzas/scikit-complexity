# Design

## Context

The parts of the framework that this change builds on:
- Elements declare states and parameters, and interactions bind elements through roles and return laws.
- The law kinds decide the form. Rate laws give an ODE and update laws give a map. Subsystems whose forms differ
  co-simulate, and other partitions are read as inputs held between synchronisation points.
- Fields put one copy of an element's laws on every lattice site.
- `Call` evaluates a Python or Numba function inside a law, and drivers and live parameter changes reach laws between
  frames.
- `biology/systems_biology` already has Boolean gene networks built from `Update` laws.

Nothing in the core is a decision variable fixed by an optimisation at a point in time, and that is what all seven
models need. Raman's static optimisation approach (Eqs. 9.8–9.9) solves the LP at `t_k`, holds the fluxes, and
integrates the exchange ODEs over `Δt`. This is the co-simulation contract with a new kind of partition.

## Goals / Non-Goals

**Goals:**

- Declare a constraint network with the framework's symbols and solve it, in the default install, under the MIT
  licence.
- Couple the solve to ODEs, maps, co-simulation, and lattice fields without new runtime machinery. The solve becomes a
  law kind with a form.
- Take the established analyses, model formats, strain-design algorithms, and community methods from COBRApy,
  StrainDesign, and MICOM through an optional extra, so the core holds only the adapters.
- Keep the core domain-agnostic. Pools, processes, and switches mean metabolites, reactions, and genes in metabolism,
  and goods, routes, and facilities in a supply chain.

**Non-Goals:**

- Nonlinear constraint-based methods: thermodynamic FBA, enzyme-constrained models with nonlinear kinetics, and
  resource balance analysis. They can follow on the same network once a nonlinear program solver is adopted.
- Flux sampling (`hopsy`, OptGP). It is a separate analysis and can be added to the extra later.
- Gap filling and model reconstruction (CarveMe, ModelSEED). Models are imported, not built.
- Commercial solvers. COBRApy can still use them when the user installs one, but no test depends on them.
- Shipping genome-scale model files in the package. The E. coli core model is the `textbook` model that COBRApy
  bundles, and the other BiGG models are downloaded on request.

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

### Licensing: highspy in the core, COBRApy behind an extra

The project is MIT. COBRApy is dual LGPL-2.0+ or GPL-2.0+, and it always installs `swiglpk`, which is GPL-3.0.
- The default install SHALL hold no copyleft package, so COBRApy, StrainDesign, and MICOM go into an optional extra:
  `[project.optional-dependencies] metabolism = ["cobra>=0.32.1", "straindesign>=1.19.1", "micom>=0.39.1"]`.
  StrainDesign and MICOM are Apache-2.0, but both require `cobra`, so they share the extra.
- Only `skcomplex.core.backends._cobra` imports them, lazily inside its functions. This is the one lazy import the
  constitution allows, with a suppression whose reason names the extra.
- When the import fails, the adapter raises `MissingExtraError`, a `BackendError` defined in `skcomplex.core.base`.
  Its message names the function, the extra, and the command `pip install 'scikit-complexity[metabolism]'`, and says
  that `ConstraintNetwork` solves, `fseof`, and `steadycom` run without the extra.
- The package uses COBRApy under the LGPL option and never imports `swiglpk` itself. The user guide states that
  installing the extra brings GPL-3.0 code (GLPK) into the environment. A redistributed environment with the extra is
  then governed by that licence, but the MIT source is not.
- deptry learns that the three packages are optional imports.

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
from skcomplex.core.model import ConstraintNetwork, Constraint, Process, Symbols, either, both, total_flux

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
- A switch is a parameter with the domain `{0, 1}`, default 1. A rule is a Boolean expression in `either`, `both`, and
  switch names, the same functions laws already use. When its rule is false, a process's bounds become 0, which is
  COBRApy's gene-reaction semantics.
- A knock-out sets a switch to 0. This works as a parameter at construction, as a live parameter change, or as a
  `Switch` law.
- `Constraint(expression, lower=None, upper=None)` adds a row over flux symbols, such as an enzyme capacity or a
  maintenance demand.
- `objectives` is a sequence of `(expression or process name, 'max' | 'min')`. The solve optimises them in order, and
  fixes each optimum within `optimality=1e-9`, relative, before the next. `total_flux()` is the sum of absolute fluxes,
  which the lowering splits into the positive and negative parts of reversible processes.
- `infeasible='raise' | 'zero' | 'hold'` picks what an infeasible solve does: raise `InfeasibleError`, set all fluxes
  and the objective to 0, or keep the last fluxes.
- `space=None | LatticeSpace | NetworkSpace` copies the network to every site.
- Construction validates the declaration against a schema. It rejects an unknown pool or switch, a lower bound above
  the upper bound, a rule over undeclared switches, and an objective that is not linear, naming the process or the
  constraint.

`ConstraintNetwork` is an `Element` built without subclassing, so it stores its arguments unmodified. Its derived state
is `stoichiometry_`, a SciPy sparse matrix, and `processes_`, `pools_`, and `switches_`.

Alternatives considered:
- A table, a matrix and bound vectors, as the only input. It is kept as `ConstraintNetwork.from_arrays(S, lower,
  upper, ...)`, but symbols are the primary form, so bounds and rules can be expressions.
- Reaction strings like COBRApy's. They are metabolism-specific, and import covers files.

### Bound, switch, and solve laws

- `Bound(flux, lower=None, upper=None)` is a law an interaction returns. Its expressions may read any state, parameter,
  or input. At each solve, the effective bounds are the declared bounds intersected with every `Bound` on that flux:
  the largest lower bound and the smallest upper bound.
- `Switch(switch, value)` sets a switch from an expression, such as the state of a Boolean node. A value above 0.5
  counts as on.
- `Solve(network)` is the law each network contributes. The user never writes it.

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

The form is inferred from the law kinds, with `Bound` and `Switch` treated like `Boundary`, never changing the form:

| Law kinds in one subsystem | Form | Meaning |
|---|---|---|
| `Solve` | constraint | `ConstraintStepper` solves once per frame with the current parameters and drivers, so a parameter sweep makes a phenotype phase plane |
| `Solve` and `Rate` | ODE | the drift solves first at each evaluation, caching on `(t, y)`, then evaluates the rates that read fluxes. This is the direct approach |
| `Solve` and `Update` | map | each step solves with the switches of the last step, then applies the updates |
| `Solve` in a sibling subsystem | co-simulation | the constraint partition solves at each synchronisation point, and other partitions read its fluxes as held inputs. This is the static optimisation approach, exact for the exchange ODEs over each interval |

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

A network with `space=lattice` lowers to one `ConstraintProblem` and an array of bounds per site. Interactions with
fields bind per site, as field laws do. In COMETS, uptake bounds are Michaelis-Menten in the site's metabolite
concentrations, biomass grows by its growth flux, and metabolites and biomass diffuse with `laplacian`.
- `ConstraintNetwork(active=expression)` skips the solve at sites where the expression is false, such as biomass below
  a threshold, and gives those sites zero fluxes. This is COMETS's own rule, and it keeps the cost proportional to the
  colony, not the lattice.
- Each active site warm-starts from its own last basis, kept per site, and solves run in a loop over active sites.
- The partition solves at synchronisation points, and the field partition integrates diffusion and exchange with the
  held fluxes.

Alternative: cometspy drives the Java engine and is GPL-3.0. It cannot run the framework's fields or solvers, and it
adds a JVM.

### Regulatory FBA

Regulation is a subsystem of Boolean nodes with `Update` laws, as in the library's gene networks. An interaction
returns `Switch` laws that set the network's switches from the nodes, and `Update` laws that set a node's inputs from
the external metabolite states and the fluxes, as in Covert et al. (2001). Raman's Eq. 9.7 alternates the regulatory
step and the solve. It is the co-simulation of a map partition, a constraint partition, and an ODE partition for the
medium, with Gauss-Seidel order and the regulatory time step as the synchronisation interval.

### Static analyses through the extra

The private module `skcomplex.core.backends._cobra` holds the adapters. Each takes a network or a system holding one,
and a solution or settings, and returns a tidy `pandas` result indexed by process, switch, or design. These names are
public in `skcomplex.core.backends`:
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

These live in `backends`, because the layout reserves that package for translating models into third-party libraries
and puts it above every other layer.

A parity test compares the core solve with COBRApy's `optimize` on the toy network and on E. coli core within 1e-7.
It compares the core's `total_flux` objective with `pfba` the same way, so the in-house lexicographic objective cannot
drift from the reference.

### In-house analyses on the core solve

`skcomplex.core.analysis` gains three functions, which need no extra:
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

### Results, observables, and trajectories

Fluxes, objective values, and switch values are recorded per frame under dotted names:
- `cell.flux.convert`, `cell.objective.0`, and `cell.switch.g1`;
- per site, as arrays shaped like the lattice.

Pool duals are recorded on request with `record_duals=True`, so dense duals do not bloat long recordings. Observables
reduce over fluxes like over states.

### Visuals

`FluxVisual` is registered as `'fluxes'` and is the default visual of a non-spatial network. It logs:
- a `rerun.BarChart` of the fluxes in process order under `<path>/fluxes`, with an optional `processes=` subset for
  genome-scale models;
- the objective values as scalars under `<path>/objective`;
- with `graph=True`, the default for at most 100 processes, a `rerun.GraphNodes` of pools and processes and a
  `rerun.GraphEdges` from the coefficients, with edge widths proportional to `|flux|` and colours by sign. The node
  layout is computed once with `networkx.spring_layout`, seeded, so captures are reproducible.

A network on a lattice defaults to `ImageVisual` of a chosen flux per site, `growth` by default, beside the field
images. Rerun 0.39 has all three archetypes, which the spike checked.

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

## Migration Plan

The change only adds things:
- `highspy` joins the runtime dependencies, and the Project Profile's dependency list grows by it in the same PR;
- the `metabolism` extra is added;
- the nox `tests`, `docs`, and `gallery` sessions install the extra;
- one CI job runs a default install without the extra and checks the missing-extra errors.

The seven inventory entries move from blocked to pending this change, and to implemented as their models land.

## Open Questions

- None. The toy network's optimum is in Raman section 8.6: with the bounds of the example (internal fluxes in [0, 1000],
  reversible ones in [-1000, 1000]), growth is 50 (Eq. 8.9), and FVA gives the ranges of Eq. 8.12.
