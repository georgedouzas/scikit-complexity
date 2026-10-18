# Design

## Context

SymPy is the symbolic layer today. Symbols are `sympy.Symbol`, vector states are `sympy.Matrix`, the boxes use
`sympy.Piecewise` and `sympy.Mod`, kernels lower through `sympy.lambdify` with `cse=True`, the Numba backend prints code
with `PythonCodePrinter` and `sympy.cse`, the LAMMPS backend prints Lepton with a `StrPrinter` subclass, and the
analysis layer solves ODEs with `sympy.dsolve`. Sage enters only through the `sage` extra and `sympify`.

Measured on the current environment with passagemath 10.8.13: `passagemath-symbolics` imports in about 1.5 seconds,
depends on SymPy itself, ships wheels for Linux and macOS 13 or later but not Windows, and needs NumPy 2.1 or later.
`desolve` fails without `passagemath-maxima`, which brings ECL and has wheels on the same platforms.

## Goals / Non-Goals

**Goals:**

- One symbolic language, Sage, in the public API and inside the framework.
- Identical numeric results for every library model, so the change is a replacement of the layer, not of the physics.
- No change to agent rules, callables, co-simulation, or the runtime.

**Non-Goals:**

- Accepting SymPy expressions through conversion. The user asked for a replacement, and two input languages keep the
  documentation and the error surface doubled.
- Removing SymPy from the environment, which passagemath requires.
- Using Sage features beyond expressions and ODE solving, such as exact linear algebra in the analysis layer. They can
  follow once Sage is mandatory.

## Decisions

### Dependencies and platforms

`passagemath-symbolics`, `passagemath-maxima`, and `cysignals` become runtime dependencies, `sympy` leaves the
dependency list, the `sage` extra is removed, and NumPy's floor rises to 2.1. The framework imports Sage once, through
`sage.all__sagemath_symbolics`, in a private `core` module that every layer imports from, so the import cost is paid
once and the import path lives in one place. Windows is dropped from the supported platforms, the `LAMMPS on Windows`
workflow is removed, and the Project Profile records Linux and macOS 13 or later.

- Alternative considered: import Sage lazily on first use. Rejected because every component builds symbols when its
  laws are collected, so nearly every run pays the cost anyway and the laziness only moves it.

### Symbols and vectors

Every state, parameter, and time symbol is created with `SR.symbol` under its unique generated name and declared real
or integer at the Pynac level with `decl_assume`, and `core` records its full `Domain` by name. A Sage domain given to
`SR.var` would make Sage record it in Maxima, which loads Maxima on first use, about 0.6 seconds that would break the
cached compilation requirement. `declare_domains` gives Maxima the full domains, positivity included, before an exact
solution. Vector states become `vector(SR, ...)`, which keeps `dot`, `norm`, indexing, and arithmetic.
Matching symbols to layout entries uses Sage's structural equality instead of SymPy's hashing.

### Conditions and remainders

Sage's `piecewise` is defined over domains of one variable and cannot express a condition on several symbols, and SR
has no symbolic remainder. `skcomplex.model` adds `where(condition, then, otherwise)`, `either(*conditions)`,
`both(*conditions)`, and `mod(value, divisor)`, each a Sage `BuiltinFunction` that stays unevaluated on symbolic
arguments and evaluates on numbers. Conditions are Sage relations such as `x < 0`. The printers lower them to
`numpy.where` and `numpy.mod` for NumPy, to conditional expressions and `math.fmod` with sign correction for Numba, and
to `select` and `step` for Lepton.

- Alternative considered: Heaviside products such as `heaviside(x) * a + (1 - heaviside(x)) * b`. Rejected because
  they evaluate both branches, give `nan` when a branch is infinite, and hide intent in the equations users read.

### One printer for every target

A private `compile` module walks Sage expressions through `operator()` and `operands()` and prints them with a target
table: NumPy for kernels, scalar Python with `math` for Numba, and Lepton for LAMMPS. Common subexpression elimination
counts repeated subtrees over all the expressions of a kernel, keyed by their structural hash, and binds each subtree
that occurs at least twice and is not a symbol or number to a numbered local. NumPy kernels are compiled from the
printed source with `exec`, as `lambdify` did, and the Numba backend keeps its source-keyed disk cache.

- Alternative considered: Sage's `fast_callable`. Rejected because it evaluates one point at a time, cannot be
  vectorized over group members, and produces nothing Numba or LAMMPS can consume.
- Alternative considered: convert to SymPy inside lowering and keep `lambdify`, `cse`, and the printers. Rejected
  because it keeps SymPy as a hidden layer, which is what the change removes, and adds a conversion of every expression.

### Rejecting SymPy

Lowering accepts Sage expressions and vectors, Python and NumPy numbers, and callables. Any other value raises
`ValidationError` with the component path and the law, and when the value has a `_sage_` method the message names it,
`expression._sage_()`, as the conversion to use. A `ruff` banned-API rule forbids importing `sympy` in `src`.

### Exact solutions

`equations_of` returns relations `diff(n(t), t) == rhs` for continuous forms and `n(t + 1) == rhs` for discrete maps,
with every state a symbolic function of `t`. `solve_ode` uses `desolve` for one equation and `desolve_system` for
several, which handles linear systems with constant coefficients through Laplace transforms, and passes the initial
state of the system as initial conditions at `t = 0`. Equations Maxima cannot solve raise `AnalysisError`, as today.

### Equivalence with the current results

Before any code changes, a script records the trajectories of every library model and of the user guide examples with
fixed random states, and a test compares the new trajectories against them within a relative tolerance of 1e-10. The
recorded data is removed once the change is archived.

## Risks / Trade-offs

- [No Windows support] → Recorded as a breaking change in the changelog and the Project Profile. Users on Windows run
  the framework in WSL.
- [Import time grows by about 1.5 seconds] → Paid once per process. The real-time clock starts after compilation, so
  simulations are not slowed.
- [Maxima solves fewer nonlinear ODEs than `dsolve`] → The specified scenarios are linear. Unsolved systems keep the
  named error, which points to numeric simulation.
- [Sage's simplification reorders terms, so printed equations change] → Specs and tests compare expressions by
  structural or numeric equality, not by their text.
- [A new printer can mis-lower an operator] → Every operator the library uses has a printer test per target, and the
  recorded trajectories cover the composition.

## Migration Plan

User models replace `sympy` functions with their Sage equivalents, such as `sqrt`, `exp`, and `sin` from the framework
or from Sage, `sympy.Piecewise` with `where`, and `sympy.Mod` with `mod`. The user guide gains a migration table. There
is no compatibility shim, and the release is a major version.

## Implementation Notes

- Tables of expressions in the printer compare keys with `is_trivially_equal`, because Sage's `==` builds a relation
  whose truth Maxima may be asked to prove, which fails on the framework's own functions.
- `desolve_system` resets the initial value of only its first function in Maxima, so `solve_symbolically` resets all
  of them after every solution, or later solutions would inherit them.
- The NumPy and Numba dialects also print `gamma`, which `lambdify` supported before.
- The equivalence baseline keys documentation outputs by their order among printing blocks, so adding a block that
  prints nothing keeps the keys.
- The changelog is generated from commit messages, so the breaking entry is the `BREAKING CHANGE` footer of the commit.
