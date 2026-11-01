# Design

## Context

- **What the core has:**
  - `TransferInteraction(label, source, target, state, into, amount, flow, legs, rate)` in its symbolic form, which
    lowers every leg to a `TransferLaw`. When source and target are one element, a leg named by `flow=` or by the
    fourth item of a leg is reported as `<element>.<flow>`, as a flow the element declares itself would be, and
    `Source()` and `Sink()` stand for the environment.
  - `RateInteraction(target, state, rate)` and `RelaxationInteraction(target, state, goal, time)`, whose goal and time
    are functions of the namespace or numbers.
  - The law utilities of `skcomplex.core.laws.utils`: `lookup`, `where`, `step`, `ramp`, `pulse`,
    `pulse_train(t, period, width, height, start, count=None)`, `delay1`, `delay3`, `smooth`, `distributed_delay`,
    `delayed`, `clip`, `minimum`, `maximum`, `total`, `uniform`, and `normal`.
  - An element type declares its states and parameters as annotated arguments of its `__init__`, with `StateSpec` and
    `ParameterSpec`, so the states of a model are fixed by a class.
  - `ObservableSpec` records a named expression at every frame, `LoopAnalysis` lists the feedback loops of the signed
    dependency graph with flows as nodes, and `Simulation(invariants=...)` checks conserved totals.
- **What PySD offers (version 3):**
  - Translators parse a file into an `AbstractModel`: `pysd.translators.vensim.vensim_file.VensimFile(path)`, then
    `.parse()` and `.get_abstract_model()`, and the same for XMILE with `XmileFile`.
  - An `AbstractModel` holds `AbstractSection`s. The main section holds `AbstractElement`s and the subscript ranges as
    `AbstractSubscriptRange`. Each element holds `AbstractComponent`s, `AbstractLookup`s, or `AbstractData`, each with
    its subscripts and an expression tree, its `ast`.
  - The tree is made of `ArithmeticStructure`, `LogicStructure`, `ReferenceStructure`, `CallStructure`, and the
    dynamic structures: `IntegStructure`, `DelayStructure`, `DelayNStructure`, `DelayFixedStructure`,
    `SmoothStructure`, `SmoothNStructure`, `LookupsStructure`, `InlineLookupsStructure`, and others.
  - Control elements carry `INITIAL TIME`, `FINAL TIME`, `TIME STEP`, and `SAVEPER`.
- **What `Simulation` lacks:** a start time. Runs start at zero.

## Goals / Non-Goals

**Goals:**
- A translated model is an ordinary model module: its system checks, compiles, runs with any ODE solver, and works
  with the analysis layer, including loops and conserved totals.
- One translation from the abstract model, so Vensim and XMILE share all the mapping code.
- The translation uses only the existing interactions and utilities, and adds none.

**Non-Goals:**
- Writing models back to Vensim or XMILE.
- Running PySD's generated code or its runtime.
- Arrays over more than one subscript range, models that mix subscripted and unsubscripted stocks, macros, data from
  spreadsheets, `GAME`, `ALLOCATE`, and the vector functions. All of them are reported as unsupported.

## Decisions

- **The translator writes source.** It returns the model as module source in `source_`, never a system built at run
  time. Decided on 2026-11-01.

### Library survey
- **PySD 3, chosen.** It is the maintained Python reader of Vensim and XMILE, MIT licensed, and its abstract model is a
  documented, language-neutral boundary.
- **Parsing Vensim and XMILE in house, rejected.** Vensim's grammar alone is the bulk of PySD, and the constitution
  takes such a capability from a mature library.
- **BPTK-Py, rejected.** It interprets XMILE with its own runtime and exposes no abstract model to translate.
- **Wrapping a PySD `Model` as an `ExternalSolver`, rejected.** The model stays opaque: it gets no checks, no loop
  analysis, and no choice of solver.

### Translate to source, not to a system object
A model's states and parameters are annotated arguments of an element class, so a system built at run time from a file
would need a class created at run time, which the project rules out. The translator therefore writes the model as the
Python source of a module, the way a modeller writes a library model:
- one element class, named after the file, such as `TeacupElement`, with every stock an annotated `StateSpec` argument
  and every constant an annotated `ParameterSpec` argument, each defaulting to its value in the file,
- one private function per auxiliary, of the element's namespace, in dependency order,
- a function `create_system()` that returns the `System` of the element and its interactions, a function
  `observables(system)` that returns one `ObservableSpec` per auxiliary and per goal-seeking flow, and the module
  constants `START`, `UNTIL`, `DT`, and `SAVE_EVERY`.

The user writes `source_` to a file and imports it, and the module doubles as an example of a stock-and-flow model
built on the general classes. The source uses only public names, so it passes the same checks as a library model.

### The translator
`SystemDynamicsTranslator(path, *, name=None)` in `skcomplex.engine.translators` derives from `BaseConfig`:
- `check()` validates that the path names a `.mdl`, `.xmile`, `.xml`, or `.stmx` file and that `name`, when given, is
  an identifier,
- `prepare()` reads the file with PySD's translator for its format and keeps the abstract model as `model_`,
- `run()` translates it, sets `source_` and `settings_`, and raises one `ValidationError` for everything that does not
  translate.

`settings_` is a frozen `RunSettingsSpec(start, until, dt, save_every)` in `skcomplex.specs`, where `until` is
`FINAL TIME - INITIAL TIME`, so the system needs no start time. The same values appear in the module constants.

### Flows by the signs in the stocks
Each stock's `IntegStructure.flow` is split into signed terms.
- A flow variable that appears as `+` in exactly one stock and `-` in exactly one other becomes
  `TransferInteraction(flow_name, source=element, target=element, state=outflow_stock, into=inflow_stock,
  amount=..., flow=flow_name)`, reported as `<element>.<flow_name>`.
- A flow variable that appears in one stock only becomes a transfer with `state=Source()` for an inflow or
  `into=Sink()` for an outflow, reported the same way.
- Terms that are not flow variables, or that appear in several stocks, form the remaining rate of their stock. When
  that rate is one term `(goal - stock) / time`, or `(goal - stock) * rate`, where neither the goal nor the time reads
  the stock, it becomes `RelaxationInteraction(target=element, state=stock, goal=..., time=...)`. Otherwise it becomes
  `RateInteraction(target=element, state=stock, rate=...)`.

So conservation and loop polarity come out of the translated model without the modeller's help, since
`LoopAnalysis` reads the flows as nodes.

### Names
Vensim names with spaces and symbols become snake-case identifiers, such as `Contact Rate` to `contact_rate`. A
collision, or a name that clashes with an argument of the element's base class, raises one named error that lists the
clashing originals.

### Time
`Time` maps to `t + START` in every expression. A run follows the settings:
`Simulation(module.create_system(), dt=module.DT, observables=module.observables(system)).run(until=module.UNTIL)`.

### Translation by structure type
A table maps each syntax structure and each built-in function to a function that returns the source of a Sage
expression, using the utilities listed in Context. A structure missing from the table is collected rather than raised,
and `run` then raises one `ValidationError` that lists every unsupported construct with its variable.

### Subscripts
A model whose stocks are all subscripted over the same single range becomes a `GroupElement` of the element class
with one member per subscript, in the order of the range. Subscripted constants become values per member, and
unsubscripted constants values shared by every member. Flows and rates run per member, `SUM(x[r!])` becomes
`total(x)` over the group, and a reference to one subscript becomes the reduction over that member. A model with both
subscripted and unsubscripted stocks is reported as unsupported, since a transfer reads only its source and target
roles.

### Dependency
PySD is imported at the top of the translator's module. It is a runtime dependency, since `skcomplex.engine` imports
it, and `skcomplex.engine` re-exports nothing, so the import costs nothing until a user imports the translators
package.

## Roadmap

The integrations after this change, each its own change with its own spec. SALib designs, QuTiP, LAMMPS, Mesa, and
co-simulation are already in the library and are not listed.

| Order | Integration | What it gives | Where it plugs in |
|---|---|---|---|
| 1 | PySD | Vensim and XMILE models as model modules | `engine/translators`, this change |
| 2 | JAX with diffrax | GPU runs, gradients, vectorised ensembles | a `BaseBackend` for JAX, solvers on diffrax |
| 3 | SBML via libRoadRunner or Tellurium | the biochemical models of BioModels | a translator in `engine/translators` |
| 4 | FMI via FMPy | Modelica, Simulink, and Dymola models as units | each unit a partition of `CoSimulationSolver` |
| 5 | xarray and pandas | trajectories as labelled series for the PyData stack | export methods of trajectories |
| 6 | SciML via diffeqpy | stiff ODEs, SDEs, jumps, and DAEs | solvers through the `ExternalSolver` contract |
| 7 | SUNDIALS | implicit and DAE solvers | solvers through the `ExternalSolver` contract |
| 8 | NetLogo via pynetlogo | the NetLogo model library | a translator, or a co-simulation partition |
| 9 | PyMC or NumPyro with ArviZ | Bayesian calibration of parameters to data | beside `CalibrationExperiment` |
| 10 | igraph or graph-tool | network processes beyond NetworkX | the network space and the network analyses |
| 11 | NetCDF or Zarr | storage of large ensembles and fields | writers of trajectories and experiment results |
| 12 | Dask or Ray | parallel ensembles and sweeps | the executor of `skcomplex.experiments` |

## Risks / Trade-offs

- [PySD's abstract model could change between versions.] → Pin a compatible range, keep all access in the one
  translation table, and test against fixtures in CI.
- [Delays: Vensim integrates delay chains with Euler steps, while `delay1`, `delay3`, and `distributed_delay` are exact
  under adaptive solvers.] → The tests compare against PySD with a small time step and a tolerance, and the user guide
  states that runs with the file's own `dt` may differ by the Euler error.
- [Heavy dependencies (xarray, pandas).] → The first task lists which of them are new to the install.
- [Flows that are a sum of several flow variables in one term, such as `INTEG(a + b - c)`.] → They are split into
  terms, and each flow variable is classified on its own.
- [A user edits the generated module.] → That is the intent: it is ordinary model code.
