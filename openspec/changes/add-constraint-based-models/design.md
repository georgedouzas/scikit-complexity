# Design

## Context

The parts of the current code this change builds on:
- A system holds elements, interactions, and nested systems. Element types declare states with `StateSpec` and
  parameters with `ParameterSpec`, and return laws from `_get_laws(s)`. Interactions bind paths through roles.
- `StateSpec(algebraic=True)`, from `add-acausal-components`, declares a state that no law integrates and whose value
  the laws of the system determine at every evaluation. The trajectory records it like any state.
- `ParameterSpec(adaptive=True, bounds=...)` declares a parameter that laws may change and that the trajectory records.
- `GroupElement(label, element=..., on='sites')` on a `LatticeSpace` holds one member per site, and
  `DiffusionInteraction` diffuses a state of a field or lattice group.
- `CoSimulationSolver` runs the partitions of nested systems whose forms differ, and every partition reads the others
  as inputs held between synchronisation points.
- `CallSpec(function, arguments)` computes a law value by a Python or Numba function. Kernels of `engine.problems`, such
  as the Monte Carlo and extremal kernels, run what one law needs inside a step.
- `PassCompiler._issues` reports `IssueSpec(code, path, message, fix)` in `report_` before compilation, and a law's
  `RequirementSpec(time=...)` gives the `law-time` issue.
- `skcomplex.analysis` holds classes only, each with `kind` settings for one family, and analyses of a model take the
  system.

No law fixes states by an optimisation at a point in time, and that is what the seven models need. Raman's static
optimisation approach (Eqs. 9.8 and 9.9) solves the program at `t_k`, holds the fluxes, and integrates the exchange
ODEs over `Δt`, which is a co-simulation of a discrete-time partition beside an ODE partition.

## Goals / Non-Goals

**Goals:**
- One general law, `LinearProgramLaw`, that sets algebraic states to an optimal solution of a linear program, solved in
  the default install under the MIT licence of HiGHS.
- Couple the solve to ODEs, maps, co-simulation, and site groups through the existing forms and solvers.
- Keep the core free of field terms: variables, constraints, objectives, and switches mean metabolites, reactions, and
  genes in metabolism, and goods, routes, and facilities in a supply chain. The metabolic vocabulary, the methods
  COBRApy, StrainDesign, and MICOM provide, and isotope labelling live in the models.
- Enumerate elementary modes and extreme pathways exactly, as an analysis.

**Non-Goals:**
- Nonlinear constraint-based methods: thermodynamic FBA, enzyme-constrained models with nonlinear kinetics, and
  resource balance analysis.
- Flux sampling, gap filling, model reconstruction, and commercial solvers.
- Shipping genome-scale model files. The E. coli core model is COBRApy's bundled `textbook` model, and BiGG models are
  downloaded on request.
- Enumerating the elementary modes of genome-scale networks. Even E. coli core has too many modes for an exact
  enumeration within minutes, and the time limit says so.
- Isotopically non-stationary 13C-MFA and correction for natural isotope abundance.
- A static form of its own. A program alone under `StaticTime` runs in the static form `FormKind.STATIC` that
  `add-static-models` adds, and a ramp of a bound or a scan of knock-outs is an `Experiment` over a `GridDesign`, not a
  trajectory axis.

## Decisions

### Library survey

Checked on 2026-10-19 against PyPI, GitHub, and spike installs on macOS arm64 with Python 3.13:
- `highspy` (HiGHS) 1.15.1 (2026-07-02), MIT, cp312 to cp314 wheels for manylinux x86_64 and aarch64 and macOS x86_64
  and arm64, very active: adopted as a runtime dependency for LP, MILP, and QP.
- COBRApy (`cobra`) 0.32.1 (2026-08-11), LGPL-2.0+ or GPL-2.0+, pure Python, whose native dependencies `swiglpk` and
  `python-libsbml` ship cp312 and cp313 wheels, active: adopted behind the extra for FBA, FVA, pFBA, MoMA, ROOM,
  deletions, and SBML and JSON I/O.
- StrainDesign 1.19.1 (2026-07-26), Apache-2.0, pure Python on `pyscipopt` 6.3.0 (MIT, bundling Apache-2.0 SCIP),
  active: adopted behind the extra for OptKnock, RobustKnock, OptCouple, and minimal cut sets.
- MICOM 0.39.1 (2026-08-11), Apache-2.0, pure Python on `osqp`, `symengine`, and `highspy`, active: adopted behind the
  extra for community models and the cooperative trade-off.
- optlang 1.9.1 and `swiglpk` 5.0.13 (GPL-3.0) come only through COBRApy and are never imported.
- cameo 0.13.6 (inactive since 2022, pins `gnomic==1.0.1`), SteadyComPy (not on PyPI, stopped in 2017), cometspy 0.6.3
  (GPL-3.0, needs the Java COMETS engine), `dfba` 0.1.8 (no wheels, inactive), and mergem 1.1.0 (low activity, MICOM
  covers communities): rejected.

The spike on the E. coli core model (72 metabolites, 95 reactions) measured aerobic growth 0.8739215, anaerobic growth
0.2116629, a pFBA total flux of 518.42209, growth 1.9e-15 after knocking out `gapA` (b1779) and 0.86316 after knocking
out `pgi` (b4025), and 7 essential genes. Solving the FBA 200 times with a changing glucose bound took 1.76 ms per solve
with `scipy.optimize.linprog(method='highs')` rebuilt each call, 0.23 ms with COBRApy's `slim_optimize`, and 0.072 ms
with `highspy` keeping the model and changing only column bounds, all at the optimum 0.4178894. ROOM returned zero
growth on GLPK after the `pgi` knock-out, and optlang's `hybrid` interface (HiGHS and OSQP) gave 0.7465 linear and
0.8474 MILP. optlang 1.9.1 rejects `'highs'` as an interface name with `highspy` 1.15.1, so only `'hybrid'` reaches
HiGHS. StrainDesign with SCIP found an OptKnock design for succinate, `{PGI, ACKr, CO2t}`, in 1.9 s, with growth 0.165
and a smallest succinate secretion at maximal growth of 9.665.

### Library survey: enumerating the cone

- `passagemath-polyhedra` 10.8.13, the polyhedra distribution of the passagemath family the project already depends on,
  with the Parma Polyhedra Library backend: adopted as a runtime dependency, since the analysis layer uses it. Its
  wheels follow those of `passagemath-symbolics`, so the supported platforms stay the same.
- `pplpy` 0.9.0 (GPL-3.0+): the same library without the Sage layer, rejected because its macOS wheels need macOS 15,
  which would drop supported platforms from the default install.
- `pycddlib` 3.0.2 (no Linux or macOS wheels), `efmtool` 0.2.1 and `efmtool_link` (need a JVM), efmlrs 2.3.6
  (uninstallable on 3.12), ecmtool 0.1.12 (conversion modes, on `pycddlib`), and `polytope` 0.2.5 (no exact ray
  enumeration): rejected.

The spike on `pplpy`, which `passagemath-polyhedra` wraps, gave the 2 elementary modes of Raman's toy network of Table
8.2 in 0.2 ms, both with `vbio = 1/12` and `v6 = 5/3` normalised to `b1 = 1`, and the 20 extreme rays of the circulation
cone of the complete directed graph on 4 nodes, the 20 simple cycles of `networkx.simple_cycles`. E. coli core did not
finish within 4 minutes.

### Library survey: labelling simulation and flux estimation

- FreeFlux 0.3.8 (pins `numpy<1.23` and `scipy<1.9`), 13CFLUX v3 `x3cflux` 3.1.2 (AGPL-3.0, needs `cyipopt` from source,
  no aarch64 Linux or Intel macOS wheels), influx_si 7.4.3 (needs R), and INCA, Metran, and OpenFLUX (MATLAB or
  proprietary): rejected.
- mfapy 0.6.3 (MIT, not on PyPI, installs from GitHub): the parity reference in the test group. Its TCA example of
  Antoniewicz et al. (2007) gave the published glutamate distribution `[0.3464, 0.2695, 0.2708, 0.0807, 0.0286,
  0.0039]`, and a fit to noise-free data from two parallel tracers recovered every net flux with a residual of 8e-32,
  while the data did not identify the fumarase exchange.

So the elementary metabolite unit (EMU) algorithm and the fit are written in house, thin over SciPy's sparse LU and
`least_squares`, in the one model package that uses them.

### Licensing

The project is MIT. A library that `skcomplex.core`, `skcomplex.engine`, or `skcomplex.analysis` uses is a runtime
dependency whatever its licence, so `highspy` (MIT) and `passagemath-polyhedra` (GPL, like the passagemath distributions
already installed) join the runtime dependencies. COBRApy, StrainDesign, and MICOM are used by the models alone, so
they form the optional extra `metabolism`. Only `skcomplex.models.biology.common.metabolism` imports them, lazily, with
the one suppression the constitution allows. A missing extra raises `MissingExtraError`, which names the class, the
extra, and the command `pip install 'scikit-complexity[metabolism]'`. The user guide states that the extra brings
GPL-3.0 GLPK into the environment through COBRApy.

Can `highspy` alone replace COBRApy? It suffices for the numerics of one solve, it was 25 times faster than `linprog`
and 3 times faster than COBRApy on repeated solves, and the dynamic and lattice models need exactly that. So the core
solves programs with `highspy`. Every named analysis that COBRApy implements stays in COBRApy, reached by converting the
network, since the constitution forbids reimplementing what a mature library provides.

### The linear program law

`LinearProgramLaw(variables, constraints, objectives=(), active=None, infeasible='raise', duals=False, label='')` in
`core.laws`:
- `variables` are the algebraic states the program sets, read from the namespace.
- `constraints` are Sage relations, `==`, `<=`, or `>=`, linear in the variables with constant coefficients. Their
  other terms may read any state, parameter, or the time, such as an uptake bound that falls with a concentration. A
  relation in one variable is a bound of that variable.
- `objectives` are `ObjectiveSpec(expression, sense='max')` of `skcomplex.specs`, linear in the variables or a sum of
  their absolute values, which the lowering splits into positive and negative parts. They are solved in order, each
  optimum fixed within a relative `1e-9` before the next.
- `active` is an expression, or `None` for always, where false the solve is skipped and the variables are zero.
- `infeasible` is `'raise'`, `'zero'`, or `'hold'`: raise `InfeasibleError`, set the variables and objectives to zero,
  or keep the last values.
- `duals` records the dual of every constraint.

It requires continuous, discrete, or static time. Switches need no law of their own: a switch is an adaptive parameter
with the domain `{0, 1}`, and a gene rule is an expression such as `upper * either(s.g1, s.g2)` in a bound, written with
the utilities `either` and `both` that laws already use. A knock-out sets the parameter to 0 at construction, between
runs, or by an `UpdateLaw` of an interaction that reads a Boolean node.

### Programs that share variables

The `LinearProgramLaw`s whose variables overlap, from any element or interaction, form one program: their variables
unite, their constraints add, and at most one of them holds objectives. So an uptake interaction returns
`LinearProgramLaw(variables=[s.cell.v_glc], constraints=[s.cell.v_glc >= -s.vmax * s.medium.glc / (s.km +
s.medium.glc)])`, and joint community FBA is the members' laws merged through the exchange variables of a shared medium
element whose law holds the abundance-weighted objective. This is the same rule as the summed equations of one target.

### Lowering and solving

A pass of `engine.compile.passes` merges the laws, differentiates every relation by every variable to extract the
sparse constant matrix, and compiles the right sides, the bounds, and the activity as functions on the NumPy and Numba
backends. A kernel of `engine.problems` holds one `highspy.Highs` model per program and per member. Before each solve
it changes only the bounds with `changeColsBounds` and `changeRowsBounds`, so HiGHS starts from the last basis, and it
asks for the irreducible infeasible subset of an infeasible program under `'raise'`, which `InfeasibleError` names with
the path, the time, the member, and the constraints and variables of the subset.

The program solves where its variables are read:

- Beside rate laws under continuous time, the system keeps the ODE form, and the drift solves the program first at
  each evaluation, cached on `(t, y)`. This is the direct approach.
- Alone under static time, the static form of `add-static-models` solves the program once per frame from the current
  parameters.
- Alone or beside update laws under discrete time, the system keeps the map form, and each step solves with the values
  of the last step before it applies the updates.
- In a nested system beside a sibling of another form, the program's discrete-time partition solves at each
  synchronisation point, and the sibling reads its variables as held inputs. This is the static optimisation approach.

A program in a group on sites solves per member where `active` holds, each member with its own warm start. The direct
approach with an adaptive integrator meets a right side that is only piecewise smooth, so the library models use fixed
steps or the static optimisation approach, and the user guide says so.

### Issues

`PassCompiler._issues` adds:
- `linear-program-form`: a relation is not linear in the variables, or a coefficient of a variable reads a state. The
  issue names the relation and the variable, and the fix is to move the term to the right side or to a bound.
- `linear-program-objectives`: two merged laws hold objectives. The issue names both paths, and the fix is to keep the
  objectives in one law.

A program under any other time gives the existing `law-time` issue.

### Recording and visuals

The variables are recorded under their paths, `cell.v_growth`, the objective values under `cell.objective_<k>`, and the
duals under `cell.dual_<k>` when `duals` is true. `LinearProgramVisual` in `engine.viz` logs a `rerun.BarChart` of the
variables, optionally a subset for genome-scale models, the objective values as scalars, and with `graph=True`, the
default up to 100 variables, a `rerun.GraphNodes` of constraints and variables and `rerun.GraphEdges` from the
coefficients, with widths proportional to `|value|`, laid out once by a seeded `networkx.spring_layout`. A program in a
site group draws a chosen variable per site with the existing `ImageVisual`.

### Elementary modes

`ElementaryModeAnalysis(kind='modes' | 'pathways', normalize=None, exchanges=None, timeout=60.0)` in
`skcomplex.analysis` takes a system and reads the program of its linear program law at the initial values, or a matrix
with the indices of its irreversible columns:
- The directions come from the bounds: a negative lower bound makes a variable reversible, a nonpositive upper bound
  reverses its column, and a variable fixed at zero, such as a knocked-out reaction, is removed.
- `'modes'` splits every reversible column, takes the extreme rays of the split cone with
  `Polyhedron(eqns=..., ieqs=..., backend='ppl')`, drops the two-cycles of split pairs, and maps the rest back, which
  gives exactly the elementary modes (Gagneur and Klamt 2004).
- `'pathways'` splits only the reversible internal columns and keeps each exchange as one column in its declared
  direction (Schilling et al. 2000). A variable in one equality is an exchange unless `exchanges` names them. The lines
  of a cone that is not pointed are kept apart.
- Coefficients become fractions with denominators of at most 10⁶, and a coefficient the conversion moves by more than
  1e-9 raises a named error naming the variable and the constraint. A nonzero right side raises a named error, since
  modes are defined at steady state.
- The enumeration runs under a `cysignals` alarm. When it fires, `EnumerationLimitError`, an `AnalysisError`, names the
  number of variables and the limit, and suggests closing exchanges or enumerating a subnetwork.

It sets `modes_`, one row per mode scaled to `normalize` or to the smallest integer vector, `labels_`, the variables,
and `lines_`.

### The metabolism family

`skcomplex.models.biology.common.metabolism` is shared by `systems_biology` and `ecology`:
- `MetabolicNetworkElement`, an element type whose subclasses declare one algebraic state per reaction and one adaptive
  switch per gene, and give their reactions as `ReactionSpec(coefficients, lower, upper, rule)`, a frozen spec of the
  package. It returns one `LinearProgramLaw` with a balance per metabolite and the bounds gated by the rules.
- `from_cobra(model)` builds such a subclass from a COBRApy model, as the surrogate analyses build their element types,
  and `to_cobra(element)` converts back, with switches as genes and rules as GPR strings and the `'hybrid'` solver.
  `load_network(path_or_id)` and `save_network(element, path)` read and write SBML and BiGG JSON through `cobra.io`, and
  `'textbook'` is the bundled E. coli core model.
- `CobraAnalysis(kind='variability' | 'parsimonious' | 'deletions' | 'moma' | 'room' | 'synthetic-lethals')`,
  `StrainDesignAnalysis(kind='optknock' | 'robustknock' | 'optcouple' | 'mcs')`, and the `'tradeoff'` kind of
  `CommunityAnalysis` need the extra.
- `ProductionScanAnalysis`, FSEOF, and the `'steadycom'` kind of `CommunityAnalysis`, after Chan et al. (2017) by a
  bisection on the common growth rate, run on the core solve without the extra, since no maintained library implements
  them.

The analyses derive from `BaseAnalysis`, take the system, and set their results as attributes indexed by reaction or
gene name. A parity test compares the core solve with COBRApy's `optimize` and `pfba` on the toy network and on E. coli
core within 1e-7.

### Isotope labelling

Only `carbon_13_mfa` uses labelling, so it lives in `skcomplex.models.biology.systems_biology`:
- `AtomMapSpec(reactants, products)` pairs every occurrence of a metabolite with atom letters, `TracerSpec(metabolite,
  fractions)` gives isotopomer fractions as bit strings, and `FragmentSpec(metabolite, atoms, name)` names a measured
  fragment. They are frozen specs of the package, checked for unbalanced letters and inconsistent atom counts.
- `LabellingAnalysis(kind='simulation' | 'estimation', atoms, tracers, measurements, fluxes=None, symmetric=(),
  starts=5, intervals='linear', random_state=None)` takes the system of a metabolic network. It walks the atom maps
  backward from the fragments to collect the elementary metabolite units (Antoniewicz et al. 2007), builds per unit size
  the sparse system `A_n(v) X_n = B_n(v) Y_n`, and solves the sizes in order with `scipy.sparse.linalg.splu`.
- `'simulation'` sets `distributions_` for given fluxes. `'estimation'` parametrises the fluxes by a null space basis of
  the balances plus one exchange flux per reversible reaction, minimises the weighted residuals with
  `scipy.optimize.least_squares` from seeded feasible starts that the core solve draws, and sets `fluxes_`,
  `intervals_` (linearised, or profiled as in Antoniewicz et al. 2006), `identifiable_`, `residual_`, `chi2_`, and
  `accepted_`.

A test compares the simulation with mfapy on the TCA example within 1e-10 and with the published distribution within
5e-5.

### Library models

Every model has a gallery example with its Rerun primary view as the thumbnail and at least one Matplotlib analysis
plot, within the budget of 300 frames, 20 MB, and 60 s:
- `flux_balance`: Raman's toy network by default, or E. coli core with the extra, under static time, with the oxygen
  bound swept by an `Experiment` over a `GridDesign`. The view is the bar chart, and the plot shows the FVA ranges with
  the pFBA fluxes and growth against oxygen.
- `metabolic_perturbations`: E. coli core under static time, with the deletion scan of `CobraAnalysis` and FBA, MoMA,
  and ROOM after `pgi`.
- `regulatory_fba`: E. coli core with Covert-style rules, Boolean nodes and the program in a discrete-time partition
  beside the medium's ODE partition, the switches set by `UpdateLaw`s from the nodes.
- `dynamic_fba`: E. coli core in batch with Michaelis-Menten glucose uptake by the static optimisation approach, the
  direct approach as an option.
- `dynamic_multispecies_fba`: two E. coli core strains that differ in uptake `vmax` in one medium.
- `comets`: a 25 by 25 lattice with a program per site, active above a biomass threshold, and diffusing glucose, solved
  by the direct approach with fixed steps. At about 100 active sites, 200 frames, and four solves of 0.07 ms per step,
  the solves take about 6 s.
- `community_fba`: two E. coli core members with complementary knock-outs, by merged programs, SteadyCom, and the
  trade-off.
- `elementary_flux_modes`: the modes of the toy network by `ElementaryModeAnalysis`, drawn as a heat map and as the
  decomposition of the FBA optimum.
- `carbon_13_mfa`: the TCA example of Antoniewicz et al. (2007) with two parallel tracers, data from known fluxes with
  noise of standard deviation 0.002 at seed 0, and 5 starts, drawn as estimated fluxes with intervals against the true
  ones and as measured against simulated glutamate distributions.

## Risks / Trade-offs

- [COBRApy 0.32.1 caps `pandas<3.0`] → Mesa 3.5 accepts pandas 2.3, so the lock resolves pandas 2.3, and the first task
  checks the full suite on it.
- [GPL-3.0 GLPK comes with the extra] → No package module imports it, COBRApy is pointed to `hybrid`, and the user guide
  names the licence.
- [Non-unique optimal fluxes make dynamic trajectories depend on the solver] → The dynamic models minimise the total
  absolute flux as a second objective.
- [A depleted substrate makes the program infeasible] → The dynamic models set `infeasible='zero'`, and growth stops as
  in Raman §9.6.
- [Downloading BiGG models in tests is flaky] → Tests use the bundled model, and one test that downloads iNJ661 is
  marked `network`.
- [The number of elementary modes grows combinatorially] → The time limit with a named error.
- [The EMU code and the fit are in house] → Parity tests against mfapy and the published distribution.

## Migration Plan

The change only adds things. `MissingExtraError` in `skcomplex.errors` is added by whichever of
`add-molecular-structures` and this change lands first, and the other reuses it. `highspy` and `passagemath-polyhedra`
join the runtime dependencies and the Project Profile, the `metabolism` extra is added and installed by the `tests`,
`docs`, and `gallery` sessions, and one CI job runs a default install without the extra. The nine inventory entries move
to implemented as their models land.

## Open Questions

- Whether to add an `x3cflux` adapter once its wheels cover the supported platforms.
