# Proposal

## Why

The engine runs only the forms it knows by name.
- `compile_system` and `PassCompiler` list the quantum and rewriting compilers.
- The default solvers map every form to a solver in one fixed table.
- The backends and the co-simulation solver list the rewriting laws.

So nobody can add a new kind of dynamics without editing the engine. For the same reason, word and patch rewriting sit in the core and the engine, although one model file uses them.

## What Changes

- **A simulation takes its compiler.**
  - `Simulation` and `compile_system` take a `compiler` setting, which defaults to `PassCompiler()`. They pass their backend, solver, and observables to it.
  - A compiler sets `solver_`, the default solver of the problem it compiled, through the hook `_get_solver(problem)`.
  - **BREAKING** No compiler claims a system anymore: `claims_`, `_claims`, and the claim loop are removed. A quantum system runs with `compiler=QuantumCompiler()`.
- **No form is listed by name outside its compiler and solver.**
  - The rewriting form leaves `FormKind`. A plug-in form is a string that its compiler sets and its solver declares.
  - The backends and the co-simulation solver stop listing the rewriting laws.
- **Rewriting moves to the fractal models.**
  - `WordElement`, `PatchElement`, `ProductionLaw`, `SubstitutionLaw`, `ModuleSpec`, `PrototileSpec`, `AffineSpec`, and `GeometrySpec` move to `skcomplex.models.mathematics.fractals`.
  - So do the word and patch kernels, their structures, the rewriting problem, compiler, solver, and visual, and `RewritingError`.
  - **BREAKING** These leave the core, the engine, and the root modules.
- **Graph rewriting is deleted.** It is already done: no model used it.
- **Quantum systems use the same mechanism.** `QuantumCompiler` stays in the engine, but `PassCompiler` no longer hands quantum systems to it. The quantum models and the catalog name their compiler.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `simulation`: a simulation takes a compiler, and the compiler names the default solver.
- `rewriting-systems`: rewriting is a model package plugged in through the compiler; graph rewriting is removed.
- `package-layout`: the engine lists no form, compiler, or law by name outside their own classes.

## Impact

- **Code:**
  - `engine.compile`: `_compile.py`, `_base.py`, `_pass_compiler.py`, the quantum and rewriting compilers;
  - `engine.runtime._simulations`, `engine.solvers._defaults` and `_cosimulation`, the NumPy and Numba backends;
  - `core.elements` and `core.laws`, the specs package, `types.FormKind`, and `errors`;
  - the model catalog, which gains a `compiler` per model;
  - the fractal models and the quantum models;
  - the tests, the gallery, and the user guide.
- **API:** quantum systems need `compiler=QuantumCompiler()`, and the rewriting classes import from `skcomplex.models.mathematics.fractals`.
