# Proposal

## Why

The library defines 180 spec classes in 176 modules, 120 of them in the engine, 79 in `core/engine/problems` alone.
Most are records the compiler fills in and a solver or backend reads, and users never see them, yet 43 of the public
ones never appear in the docs. scikit-learn keeps such data as arrays, tuples and plain arguments between private
functions, uses a small private class only where many values travel together, defines it in the module that uses it,
and validates only at the public boundary. This change applies that to every internal record of the library.

## What Changes

- **Private records:** every record that only the engine, the analyses or the experiments build and read becomes a
  private frozen dataclass with no `BaseSpec` base, no `_check` and no `Spec` suffix, such as `_LatticeContext`. It is
  defined in the module of the pass, solver, backend or analysis that builds it, or of its main reader.
- **Trivial records removed:** a record with three fields or fewer, no methods and one reader becomes plain arguments
  or a tuple, such as `PrintedSpec`, `PartialsSpec`, `_SweepSpec`, `_LoopSpec` and `_InteractionRequirementSpec`.
- **Arrays for collections:** lists of small records of one kind, such as entries, terms and rows, become arrays or
  structured arrays where every member has the same fields.
- **The `problems` package holds problems:** each problem kind keeps one module with its private records, and every
  other record moves to the module that builds or reads it.
- **A small public contract:** only what a custom solver or backend reads stays public, `BaseProblemSpec` and the
  problem forms solvers receive, and their docstrings describe their fields as the attributes of a fitted object.
- **BREAKING:** public engine records that are not in the contract are no longer importable. None is in the docs.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `base-interfaces`: values and records follow the new rule.

## Impact

- `core/engine` (problems, compile, solvers, backends, translators), the private records of `core/analysis`,
  `core/components`, `core/experiments` and `systems/general/rewriting`, their tests, and the constitution.
- Must land before `simplify-user-specs`, which builds on its rule.
