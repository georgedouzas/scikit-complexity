# Design

## Context

`compile_system` asks `QuantumCompiler` and then `RewritingCompiler` whether they claim a system, and falls back to `PassCompiler`. `PassCompiler` refuses a system that either of them claims. `Simulation` picks the solver from `DEFAULT_SOLVERS[problem.form]`. Rewriting is spread over:
- the core: elements, laws, and the word utilities;
- the specs: modules, prototiles, affine maps, geometries, and structures;
- the engine: kernels, compiler, solver, and visual;
- the root modules: `FormKind.REWRITING` and `RewritingError`.

## Goals / Non-Goals

**Goals:**
- A model package can add a form with its own element, laws, compiler, kernels, solver, and visual, using only the public bases.
- Word and patch rewriting are that example.

**Non-Goals:**
- Changing what rewriting or quantum dynamics compute.

## Decisions

### The compiler is a setting
`Simulation(system, compiler=None, ...)` and `compile_system(system, ..., compiler=None)` take the compiler. The default is `PassCompiler()`.

The simulation clones the compiler and sets its `backend`, `solver`, and `observables` from its own settings. So `Simulation`'s existing keywords keep their meaning, and the compiler's own values for them are replaced.

The alternative was for the elements of a system to name their compiler. It was rejected because it hides the choice, and because a quantum system is recognized by the domains of its states, not by an element class.

### The compiler names the default solver
`BaseCompiler._run` compiles the problem and sets `solver_`: the `solver` setting if one was given, otherwise the hook `_get_solver(problem)`.
- `PassCompiler` keeps the table of its own forms, with `CoSimulationSolver` for the coupled form.
- `QuantumCompiler` returns `LindbladSolver()`.
- The fractals' rewriting compiler returns its rewriting solver.

The simulation runs `compiler.solver_`.

### Forms are strings outside the core forms
`FormKind` keeps the forms that `PassCompiler` and `QuantumCompiler` produce. A plug-in form is a plain string constant of its package, such as `'rewriting system'`.
- Its compiler sets it on the problem, and its solver returns it from `_get_forms()`.
- The simulation decides whether time steps are whole numbers from the solver's discreteness, not from a list of forms. The discreteness becomes a hook of `BaseSolver`, `_is_stepwise()`.

### Laws are listed only by the classes that read them
A backend's support lists the laws it lowers, and rewriting laws are not lowered by backends. The co-simulation solver lists the laws its partitions take, and its partitions never hold rewriting laws. So both lists drop the rewriting laws.

### Where rewriting goes
All of it goes to `skcomplex.models.mathematics.fractals`, in `common`-free modules beside the models, since one package uses it.
- The elements, the laws, and the word utilities.
- The module, prototile, affine, and geometry specs, and the word and patch structures.
- The kernels, compiler, solver, problem spec, and visual.
- `RewritingError`.

The import-linter contracts already allow models to import the public modules of the core and the engine. Any private name the move needs is made public on its base, or the move is reported.

### Quantum
`PassCompiler` stops checking for quantum systems. A quantum system compiled without `QuantumCompiler` fails `PassCompiler`'s validation with an error that names `compiler=QuantumCompiler()`. The quantum models register `compiler=QuantumCompiler()` in the catalog, and `models.run` and `models.create` users pass it to `Simulation`.

## Risks / Trade-offs

- **Quantum users must now pass the compiler.** The error message names it, and the catalog carries it for the library's models.
- **A plug-in form is an untyped string.** It is a constant of its package and is compared, never parsed.
