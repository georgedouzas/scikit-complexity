# Proposal

## Why

The framework keeps two symbolic languages today: SymPy, which every law is lowered through, and Sage, an optional
extra whose expressions are converted to SymPy. One mandatory symbolic layer built on Sage gives users its faster
expression engine and its wider mathematics, removes the conversion step, and leaves a single language to learn and
document.

## What Changes

- **BREAKING** Make Sage, through the passagemath distributions for symbolics and Maxima, a runtime dependency, and
  remove the `sage` extra.
- **BREAKING** Write every symbolic law, energy, pair potential, and boundary as a Sage expression. Symbols of states,
  parameters, and roles become Sage symbols, and vector states become Sage vectors.
- **BREAKING** Reject SymPy expressions in laws with a named error that tells the user to write the expression in Sage.
- Add conditional and remainder functions for laws, because Sage has no condition-based piecewise expression or
  symbolic remainder, and use them in the reflecting and periodic boxes.
- Lower Sage expressions directly to NumPy, Numba, and LAMMPS Lepton code, with common subexpression elimination, so
  the framework no longer imports SymPy.
- Return equations as Sage relations and solve ODE systems exactly with Sage's solvers backed by Maxima.
- **BREAKING** Drop Windows, which has no passagemath wheels, and require NumPy 2.1 or later, which passagemath
  requires. Linux and macOS 13 or later stay supported.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `symbolic-analysis`: Sage becomes the only symbolic language, and SymPy expressions are rejected.
- `system-modeling`: laws express conditions and remainders through framework functions.

## Impact

- Every module that builds or lowers expressions: `model` laws and symbols, `compile` lowering, kernels, and the Numba
  backend, `solvers` steppers, `analysis`, the physics models, and the Lepton printer of `backends`.
- `pyproject.toml` dependencies, the constitution's Project Profile, the README, the user guide, every docstring
  example that builds an expression, and the tests.
- The `LAMMPS on Windows` workflow is removed with Windows support.
- Importing the framework imports Sage, which adds about 1.5 seconds.
- SymPy stays installed as a dependency of passagemath, but the framework neither imports nor accepts it.
