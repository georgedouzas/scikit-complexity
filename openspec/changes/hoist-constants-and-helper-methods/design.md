## Context

See proposal.md for the motivation. The constitution already says that a module constant is `UPPER_CASE`, lives in
the constants block near the top, and carries no leading underscore, and that comments are rare. `package-layout`
already says that every helper is a private method of the class that uses it. Neither rule names its exceptions, and
no test checks either, so the code drifted.

Scope, measured with an AST scan of `src/skcomplex` at commit `e48031c6c`. The figures are a snapshot. The four
changes this one follows move and remove many of these modules, so the first task measures again.

| What | Core | Systems |
| --- | --- | --- |
| Private module functions | about 620, in about 80 files | about 430, in about 150 files |
| Of them Numba-compiled | about 35, in 10 files | none |
| Of them read only by public functions and their helpers | about 430, nearly all in `analysis` and `experiments` | about 80 |
| Of them read by one class, or by several classes, of their module | a few | about 310 |
| Of them imported by another module | about 85, mostly `analysis/_base` and `experiments/_runs` | 5 |
| Of them read by a module value | about 15 | about 30 |
| Private module constants | about 190, two thirds in `engine/viz`, `engine/solvers`, `engine/compile` | 3 |
| Public module constants | about 680 | about 680 |
| Constants below the first definition | about 310 across both, about 30 of them built from a definition of their module | |
| Gaps of blank lines inside a run of constants | about 310 across both | |
| Constants with an attribute docstring | about 60 across both | |
| Public functions of private modules that no package re-exports | `engine/_utils.py` (9), `step_count`, `probe_laws` | `corotational`, `upper_convected`, `spin_space`, `site_space`, `velocity_of`, `landscape` |

No module defines an upper-case constant inside a function. One rename clashes: `_MEMBERS` in
`engine/compile/passes/_layout.py` beside the imported `MEMBERS`. Tests import about 16 private functions and no
private constant.

## Goals / Non-Goals

**Goals:**

- Every module of `src/skcomplex` follows the two rules, with the exceptions named here and in the constitution.
- A test in `tests/test_sources.py` keeps them true.
- Behaviour, public names, and performance do not change.

**Non-Goals:**

- `tests` and `docs`. Test functions and fixtures are module functions by design, and the gallery scripts are
  examples. See open question 4.
- Class-level constants. The constitution already makes them `ClassVar`.
- Lowercase module values that are part of the public API, such as `where = _Where()` in `laws/utils`. They are
  function objects, not constants.
- Renaming moved helpers to verb names. See open question 3.
- Any change of public names, signatures, or `__all__`.

## Decisions

### 1. Layout of the top of a module

A module reads: docstring, licence header, imports, the constants block, the type aliases, then the definitions in
the constitution's order. This keeps the user's words, constants right after the imports, and the constitution's
"module constants and type aliases". The block holds one constant per statement run with no blank line, no comment
line, and no string between them, and no comment sits right above it. A blank line inside a multi-line value counts
as a gap too, so the four values that hold one are reformatted.

An attribute docstring under a constant is a comment in effect, so it goes. What it says moves into the module
docstring, or into the docstring of the function or class that reads the constant, where the reader needs it.

A constant annotation must not name a type alias of its own module, because Python 3.12 and 3.13 evaluate module
annotations when the line runs. Three constants of `systems/biology/systems_biology/_logical_networks.py` do that
today. They drop the annotation, and mypy checks the literal where it is passed. Putting the aliases first was
considered. It would make that case free, but it moves the aliases of more modules than it saves, and it contradicts
"constants right after the imports".

### 2. Public names and the surface

Every private constant is renamed by dropping the underscore. The surface of a package is its `__all__`, and every
private constant lives in a private module, so no rename adds a public name. No renamed constant enters an `__all__`.
The scan found no private constant in a public module such as `core/types.py`. If an earlier change adds one, it
gets a public name and stays out of `__all__` too. The check compares the union of every `__all__` before and after.

A rename that clashes with a name the module already binds takes a more specific name that says what the value holds,
as the constitution asks. The one clash found becomes `HIDDEN_MEMBERS`. The import sites of the renamed constants,
about 60 of them inside the package, change in the same commit.

A mutable module registry, such as the cache of compiled Jacobians in `analysis/_lyapunov.py`, is created from a
literal and stays in the block.

### 3. Derived values

A module value built by calling a class or a function of its own module cannot sit above that definition. It is a
derived value. It keeps a public `UPPER_CASE` name and sits in one block, with no gap, right after the last
definition it reads. The cases found are the Sage symbolic functions of `laws/utils/_reductions.py`, `MITTAG_LEFFLER`,
`POSITIVE_VALUES`, `STOP_GRADIENT`, `HELPERS_FINGERPRINT`, the four dialects and `DIALECTS` of
`specs/_dialect_spec.py`, `SYSTEM_CLASSES` of the catalog, and in systems the tracer and atom map tables of
`_carbon_13_mfa.py`, the landscapes of `_optimisers.py`, the predictors of `_el_farol.py`, the plants of `_koopman.py`,
the LAMMPS tables, the collisions of `_lattice_gases.py`, `SPIN_HALF`, and the links of `_yang_mills.py`.

A function that a derived value reads, such as a Sage print hook, stays a module function beside the value. It cannot
be a method of a class that the value does not belong to, and the value must exist when the module loads.

Alternatives considered: a cached class method that builds the value on first use changes when the value exists and
its identity across imports, and adds machinery. Rewriting every Sage hook as a `BuiltinFunction` subclass, the way
`_Where` and `_StopGradient` are written, is open question 2.

### 4. Which class owns a helper

The readers of a module function are the definitions that name it, followed through other module functions. A class
reader makes it a method.

- One class reads it: that class owns it.
- Several classes of the module read it: their closest common base defined in the same module owns it. Without one,
  the first reading class in module order owns it, and the others call `Owner._name(...)`. No base is added for this,
  since the constitution asks for the fewest classes.
- Classes of other modules import it: the class the defining module exists for owns it. When the defining module holds
  no class, the reading class whose concept the helpers serve owns them and the module merges into that class's module.
  `engine/compile/_complex.py` becomes static methods of `LowerComplexPass`, and `ObservableEvaluator` and
  `ReductionLowering` call `LowerComplexPass._split_complex`. When no reader serves the concept, the helpers become
  public functions of the package `_utils` module, as the constitution asks for small general helpers. This is a move
  between modules of one package, never a package move.
- An owner must sit where every reader can import it without a cycle. When the natural owner would close a cycle,
  the helper goes to the package `_utils` module instead.

The kind of method follows what it reads. It is an instance method when it reads `self`, a `@staticmethod` when it
reads no instance state, and a `@classmethod` only when it needs `cls`, to read a class attribute a subclass overrides
or to build an instance. Inside the owner and its subclasses, instance methods call `self._name`. Static methods call
`Owner._name`, as `LayoutPass._declare_hidden` already does. Private methods come before public ones, in dependency
order, as the constitution asks. A moved function keeps its name, its leading underscore, and its one-line docstring.

A moved function passed as a callable, such as `coefficient=_get_u_diffusion` in the reaction-diffusion systems,
becomes `self._get_u_diffusion` or `Owner._get_u_diffusion`. Static methods pickle by their qualified name, so the
parallel experiments keep working.

### 5. Numba kernels

A function decorated with `njit`, `jit`, `vectorize`, `guvectorize`, or `cfunc` of `numba`, in any import form, stays
a module function. This is the exact rule, and it holds for public kernels such as `step_count` too.

The reasons come from the code. Compiled code resolves a callee as a module global when it compiles, and it cannot
look up an attribute of a plain Python class, so a compiled static method could not call another one. The kernels of
`_numba_neighbours.py` call one another this way. The generated kernels of `backends/_numba.py` import the helpers of
`_numba_components.py`, `_numba_neighbours.py`, `_numba_stencils.py`, and `problems/_steps.py` by module path, and
`NumbaBackend._helpers_fingerprint` hashes the source of those modules to key the cache. `problems/_word_kernel.py`
imports the compiled loops of `_rewriting_loops.py`.

Kernels sit after the constants and the type aliases and before the classes, which is where the constitution puts
private functions. No static method wraps a kernel, since a one-line wrapper is the kind of machinery the project
avoids. A plain Python function that only prepares the arguments of a kernel is not compiled, so it becomes a method
of its class like any other helper. Functions compiled at runtime from generated source, such as the Jacobians of
`analysis/_lyapunov.py`, are values, not definitions, and are not affected.

Rejected: a static method holding a compiled module function. It doubles every name, and a kernel that calls another
still calls the module function.

### 6. Helpers of public functions

The public API of `core/analysis`, `core/experiments`, `core/components/laws/utils`, and `engine/runtime/_simulate.py`
is functions, as the scikit-learn alignment decided, and some system modules re-export functions such as `life`. The
private helpers that only those functions read have no class to belong to.

Recommendation: a private module function that no class reads, in its module or in a module that imports it, stays a
private module function. This covers the shared readers of `analysis/_base.py` and the run helpers of
`experiments/_runs.py`, which keep their names and their imports within the package. A helper that a result class or
any other class reads becomes a method of that class.

Rejected: a private class per analysis module that holds its helpers. It is a class with no concept, the machinery the
user ruled out. Rejected for now: static methods of the result class. A result is a frozen value that the analysis
returns, and its computation would sit on a class whose job is to hold values. This is open question 1.

### 7. Public helpers that no package re-exports

A public function of a private module must be re-exported by its package, live in a `_utils` module, or be compiled.
`engine/_utils.py` already holds the engine's shared helpers and stays. `step_count` is compiled. `probe_laws` and its
private namespace helper move into `engine/_utils.py`, since passes and the replicate experiment read them, and
`compile/_probe.py` goes. The six system functions become private methods of the classes that read them, under the
owner rule. The examples in their docstrings move into the mirrored test modules, because a private method carries
only its summary line.

### 8. The automated check

Three tests join `tests/test_sources.py`, next to the test of forbidden words, and walk the modules of `src/skcomplex`
with `ast` and `tokenize`. Each failure names the path, the line, and the name.

- `test_module_constants_lead_their_module`: an upper-case module assignment must have no leading underscore. Every
  statement above it must be the docstring, an import, or a constant, unless its value names a definition of the
  module or a derived value, in which case every statement between the last such name and it must be another
  derived value. No type alias may
  sit above a constant.
- `test_constant_blocks_hold_no_gap`: no blank line, comment, or string between two constants of a block, none inside
  a constant's value, and no comment on the line above the block.
- `test_classes_own_the_functions_they_read`: build the readers of every module function within its module and in
  every module that imports it. A private function fails when a class reads it, unless it is compiled or a module
  value reads it. A public function of a private module fails when no `__all__` lists it, it is not compiled, and its
  module is not a `_utils` module.

The tests are parametrised by the packages they check, so each task group adds its package when the package passes,
and the last group replaces the list with the whole of `src/skcomplex`. Nothing is skipped and nothing is
allowlisted. The scan takes about a second.

Rejected: a ruff rule, since ruff has no plugin interface. A flake8 plugin, since it adds a second linter for one
check. The project already checks its sources in `tests/test_sources.py`.

### 9. How the refactor runs

Scripts in the session scratchpad do the mechanical work, never in the repository. They read modules with `ast`,
move the text of each function into its owner's body with one more indent, add `@staticmethod` where it reads no
instance state, and rewrite call sites to `self._name` or `Owner._name`. A second script hoists and renames
constants and removes gaps, attribute docstrings, and comments in the block. A person reviews each diff, writes the
module docstring text that replaces an attribute docstring, and resolves the cases a script flags: cycles, a clash of
a moved name with an existing method, and functions read both by a class and by a derived value.

After each package: `ruff format`, `ruff check`, `mypy`, the mirrored tests, and the doctests of the package pass.
When several agents run tests at once, each sets the Numba, OpenMP, and BLAS thread counts to 1.

## Risks / Trade-offs

- [The earlier changes move many of these modules] → This change lands last and the first task measures again.
- [A moved owner closes an import cycle] → The owner goes to the lower module or to the package `_utils` module, as
  decision 4 says, and `lint-imports` confirms it.
- [A moved method shadows an inherited method of the same name] → The script reports every clash. The moved helper
  takes a more specific name.
- [Renamed constants invalidate the Numba cache once] → The fingerprint hashes the kernel modules, so the first run
  after the change recompiles. This is expected and harmless.
- [Callables passed to parallel workers no longer pickle] → Static methods pickle by qualified name. The experiment
  tests that run in parallel confirm it.
- [Attribute docstrings carried knowledge] → Their text moves to the docstring that a reader of the constant reads.
- [A large diff is hard to review] → One commit per package, each with its checks passing.

## Migration Plan

The work lands package by package on `development`, after the four earlier changes. Each commit passes the full gate,
so any one can be reverted alone. No public name changes, so users need no migration.

## Open Questions

These are the user's to answer. Each recommendation is what the tasks assume.

1. Helpers of public functions in analysis, experiments, law utilities, and `simulate`: keep them as private module
   functions when no class reads them (recommended), or make them static methods of the module's result class?
2. Sage hooks of derived values, such as the print functions of the reductions: keep them as module functions beside
   the value (recommended), or rewrite each symbolic function as a `BuiltinFunction` subclass, the way `where` and
   `stop_gradient` are written, so the hooks become methods?
3. Moved helpers with noun names, such as `_noise` or `_richness`: keep the names (recommended, so the refactor stays
   mechanical), or rename them to verbs now, as the constitution asks of methods?
4. Should the constants rule also cover `tests` and `docs`? Recommended: not in this change.
