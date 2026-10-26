# Proposal

## Why

System dynamics has decades of published models in Vensim (`.mdl`) and XMILE (Stella, iThink, Vensim exports) files,
and modellers keep their work in those tools. The library already models stocks and flows (`TransferLaw`,
`lookup`, `distributed_delay`, `pulse`, `step`, `ramp`), checks conserved quantities, and ranks feedback
loops, but a modeller must retype every equation to use it. PySD already parses both formats into a language-neutral
abstract model, so reading that abstract model turns any of those files into a `System` without writing a parser.

## What Changes

- **Reading a model file:** `read_vensim(path)` and `read_xmile(path)` return the `System` of a system dynamics model
  and the run settings its control section declares (start time, final time, time step, saving period).
- **Translation from PySD's abstract model:** the readers call PySD's translators (`VensimFile`, `XmileFile`) and
  build the system from the `AbstractModel` they return, never from the Python code PySD generates or its runtime:
  - a stock (`INTEG`) becomes a state of one element, with its initial value;
  - a flow that leaves one stock and enters another becomes a `TransferLaw` from the first to the second, a flow
    that touches one stock becomes a `TransferLaw` with the environment, and what remains of a stock's net rate becomes
    a `RateLaw`;
  - an auxiliary becomes an expression in the laws, and an observable of the run, so the trajectory records it
    under its name;
  - a constant becomes a parameter, and a lookup becomes a `lookup` table;
  - the built-in functions map to the library's own: `IF THEN ELSE` to `where`, `STEP`, `RAMP`, `PULSE`, and
    `PULSE TRAIN` to `step`, `ramp`, `pulse`, and `pulse_train`, `DELAY1`, `DELAY3`, `DELAY N`, `SMOOTH`, and
    `SMOOTH N` to `distributed_delay`, `DELAY FIXED` to `delayed`, and the random functions to draws;
  - a variable subscripted over one range becomes a group with one member per subscript, and sums over the range
    become totals.
- **Named errors for what does not translate:** functions without a counterpart, such as `GAME`, `ALLOCATE`, the
  vector functions, and data read from spreadsheets, macros, and variables subscripted over more than one range raise
  one error that lists every unsupported construct with its variable.
- **Dependency:** PySD becomes a runtime dependency, imported lazily on the first read.
- **Roadmap of integrations.** This change is the first of a series that connects the library to the ecosystem
  around it, each a change of its own. The order and the place of each are in the design.
  - **Importing models:** PySD, this change; SBML through libRoadRunner or Tellurium; FMI units through FMPy; NetLogo
    models through pynetlogo.
  - **Solvers and backends:** JAX with diffrax; SciML through diffeqpy and ModelingToolkit; QuTiP; SUNDIALS.
  - **Data and interoperability:** xarray and pandas exports; igraph or graph-tool; NetCDF or Zarr storage.
  - **Inference, calibration, and uncertainty:** SALib; PyMC or NumPyro with ArviZ; Optuna or `scipy.optimize`.
  - **Scale-out:** Dask or Ray.

## Capabilities

### New Capabilities
- `system-dynamics-import`: reading Vensim and XMILE files into systems, the mapping of stocks, flows, auxiliaries,
  constants, lookups, built-in functions, and subscripts, the run settings of the file, and the errors for what does
  not translate.

### Modified Capabilities

## Impact

- **Code:** a new module under `skcomplex.core.system` with `read_vensim` and `read_xmile`, built only from the
  existing stocks-and-flows laws and `ElementComponent.define`.
- **Dependencies:** `pysd` in `[project] dependencies`, since library code outside `skcomplex.models` imports it, and
  its own dependencies (`parsimonious`, `xarray`, `pandas`). The import is lazy, so `import skcomplex` stays fast.
- **Tests:** small Vensim and XMILE fixtures under `tests/fixtures`, including the classic teacup and SIR models, each
  compared against PySD's own run of the same file.
- **Docs:** a user-guide section on importing models and one gallery example that imports a published model, runs it,
  and lists its feedback loops.
