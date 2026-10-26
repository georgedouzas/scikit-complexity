# Design

## Context

- **What the library has:**
  - `TransferLaw` (with `Source` and `Sink` for the environment), `RateLaw`, and
    `ElementComponent.define(name, states, parameters, laws)` in `core`;
  - helpers `lookup`, `distributed_delay`, `delayed`, `where`, `step`, `ramp`, `pulse`, `pulse_train`, `total`, and
    the draws `uniform` and `normal`;
  - `loops` and `loop_dominance` in `engine/analysis/_loops.py`, and the conservation check of `stocks-and-flows`.
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
- An imported model is an ordinary system: it validates, compiles, runs with any ODE solver, and works with the
  analysis layer, including loops and conservation.
- One translation from the abstract model, so Vensim and XMILE share all the mapping code.

**Non-Goals:**
- Writing models back to Vensim or XMILE.
- Running PySD's generated code or its runtime.
- Arrays over more than one subscript range, macros, data from spreadsheets, `GAME`, `ALLOCATE`, and the vector
  functions. All of them are reported as unsupported.

## Roadmap

The integrations after this change. Each is its own change with its own spec, since each adds a different capability.

| Order | Integration | What it gives | Where it plugs in |
|---|---|---|---|
| 1 | PySD | Vensim and XMILE models as systems | `core/system` readers, this change |
| 2 | JAX with diffrax | GPU runs, gradients for calibration and control, vectorised ensembles, in Python alone | a `BaseBackend` lowering to JAX and solvers on diffrax |
| 3 | SBML via libRoadRunner or Tellurium | the biochemical models of BioModels | readers and writers beside the system dynamics readers |
| 4 | FMI via FMPy | Modelica, Simulink, and Dymola models as units | each unit a partition of a co-simulation, and systems exported as units |
| 5 | SALib | Sobol and Morris sensitivity | designs and an analysis in `engine/experiments` |
| 6 | xarray and pandas | trajectories as labelled series for the PyData stack | export methods of `Trajectory` |
| 7 | SciML via diffeqpy and ModelingToolkit | stiff ODEs, SDEs, jumps, and DAEs | solvers through the `ExternalSolver` contract |
| 8 | QuTiP | open quantum systems, master equations, Lindblad | the quantum formalism lowering to QuTiP's solvers |
| 9 | SUNDIALS (scikits.odes, or through SciML) | implicit and DAE solvers | solvers through the `ExternalSolver` contract |
| 10 | NetLogo via pynetlogo | the NetLogo model library | a reader of NetLogo models, or a co-simulation partition |
| 11 | PyMC or NumPyro with ArviZ | Bayesian calibration of parameters to data | `engine/experiments` calibration, NumPyro beside the JAX backend |
| 12 | Optuna or `scipy.optimize` | parameter search and policy optimisation | searches over experiment designs |
| 13 | igraph or graph-tool | network processes beyond what NetworkX handles | the network space and the network analyses |
| 14 | NetCDF or Zarr | storage of large ensembles and fields | writers of trajectories and experiment results |
| 15 | Dask or Ray | parallel ensembles and sweeps across cores and clusters | the executor of `engine/experiments` |

- **Order:** PySD comes first because it is specified. JAX, SBML, FMI, SALib, and xarray come next because they widen
  what a modeller can bring in and take out, and JAX avoids a second runtime. SciML and QuTiP follow for depth in stiff
  and quantum dynamics, then the rest.
- **Dependencies:** a library that code outside `skcomplex.models` imports is a runtime dependency, imported lazily.
  Julia and NetLogo need their own runtimes, so their solvers and readers report a named error when the runtime is
  missing.

## Decisions

- **Read the abstract model, not the generated Python.** The abstract model is PySD's documented, language-neutral
  boundary. The generated code embeds PySD's runtime and its Euler integrator.
  - *Alternative:* wrap a PySD `Model` as an `ExternalSolver`. Rejected because the model stays opaque: it gets no
    validation, no loop analysis, and no choice of solver.
- **One element per model.** The reader defines one element type with `ElementComponent.define`, named after the
  file:
  - stocks become its states;
  - constants become its parameters;
  - the flows become its laws, and the auxiliaries expressions in them, with an observable for each.

  A system dynamics model shares one namespace, so one element keeps every reference local, and each subscripted
  variable becomes a group of its own element type.
- **Flows by the signs in the stocks.** Each stock's `IntegStructure.flow` is split into signed terms.
  - A flow variable that appears as `+` in exactly one stock and `-` in exactly one other becomes a
    `TransferLaw(source, sink, amount)`.
  - A flow variable that appears in one stock only becomes a transfer with `Source()` or `Sink()`.
  - Terms that are not flow variables, or that appear in several stocks, form a `RateLaw` of their stock.

  So conservation and loop polarity come out of the imported model without the modeller's help.
- **Names.** Vensim names with spaces and symbols become snake-case identifiers, such as `Contact Rate` to
  `contact_rate`. A collision raises a named error that lists the clashing originals.
- **Time.** `Time` maps to `t + start`. The settings are returned beside the system as a plain dict:
  `system, settings = read_vensim(path)`, then `Simulation(system, dt=settings['dt']).run(until=settings['until'])`.
  - Here `until` is `FINAL TIME - INITIAL TIME`, so the system needs no start time.
  - *Alternative:* a record class for the settings. Rejected so as to add no class for four numbers.
- **Translation by structure type.** A table maps each syntax structure to a function that returns a Sage
  expression. A structure missing from the table is collected rather than raised, and the reader then raises one
  `ValidationError` that lists every unsupported construct with its variable.
- **Where it lives.** A module `core/system/_sd_files.py` exports `read_vensim` and `read_xmile` from
  `skcomplex.core.system`. Library code outside `skcomplex.models` imports PySD, so PySD is a runtime dependency. It
  is imported inside the readers, so it costs nothing at import.

## Risks / Trade-offs

- [PySD's abstract model is internal-ish and could change between versions.] → Pin a compatible range, keep all
  access in the one translation table, and test against fixtures in CI.
- [Delays: Vensim integrates delay chains with Euler steps, while the library's distributed delay is exact under
  adaptive solvers.] → The tests compare against PySD with a small time step and a tolerance, and document that runs
  with the file's own `dt` may differ by the Euler error.
- [Heavy dependencies (xarray, pandas)] → They are imported only on first read. They are already transitive
  dependencies of the analysis stack, which should be confirmed during implementation.
- [Flows that are a sum of several flow variables in one term, such as `INTEG(a + b - c)`.] → They are split into
  terms, and each flow variable is classified on its own.
