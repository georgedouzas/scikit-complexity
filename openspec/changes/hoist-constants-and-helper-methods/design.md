## Context

See proposal.md for the motivation. The constitution already says that a module constant is `UPPER_CASE`, lives in
the constants block near the top, and carries no leading underscore, and that comments are rare. It says that a method
begins with a verb. `package-layout` already says that every helper is a private method of the class that uses it.
None of these rules names its exceptions, and no test checks them, so the code drifted.

Scope, measured with an AST scan at commit `e48031c6c`. The figures are a snapshot. The four changes this one follows
move and remove many of these modules, so the first task measures again.

| What | Core | Systems |
| --- | --- | --- |
| Private module functions | about 620, in about 80 files | about 430, in about 150 files |
| Of them Numba-compiled | about 35, in 10 files | none |
| Of them read only by public functions and their helpers | about 430, nearly all in `analysis` and `experiments` | about 80 |
| Of them read by one class, or by several classes, of their module | a few | about 310 |
| Of them imported by another module | about 85, mostly `analysis/_base` and `experiments/_runs` | 5 |
| Of them read by a module value | about 15, all Sage hooks | about 30, dispatch tables and predicates |
| Private module constants | about 190, two thirds in `engine/viz`, `engine/solvers`, `engine/compile` | 3 |
| Public module constants | about 680 | about 680 |
| Constants below the first definition | about 310 across both, about 30 of them built from a definition of their module | |
| Gaps of blank lines inside a run of constants | about 310 across both | |
| Constants with an attribute docstring | about 60 across both | |
| Public functions of private modules that no package re-exports | `engine/_utils.py` (9), `step_count`, `probe_laws` | `corotational`, `upper_convected`, `spin_space`, `site_space`, `velocity_of`, `landscape` |

No module defines an upper-case constant inside a function. One rename clashes: `_MEMBERS` in
`engine/compile/passes/_layout.py` beside the imported `MEMBERS`. Tests import about 16 private functions and no
private constant.

The tests and the documentation scripts hold about 1200 more constants. In `tests`, about 55 sit below a definition
and about 20 runs have gaps. In `docs`, the gallery tools, the generators, and the gallery hooks hold a few private
constants, constants below definitions or statements, and type aliases above constants, and a few examples define
constants after their first function.

The library defines its symbolic functions in two ways. Nine are `BuiltinFunction` subclasses, `where`, `either`,
`both`, `mod`, `derivative`, `lag`, `delayed`, `residence`, and `stop_gradient`. The others are built with Sage's
`function` factory: about 27 at module level in `laws/utils` and `analysis/_symbolic.py`, some of them with
`print_func`, `eval_func`, or `evalf_func` hooks, plus the delays of `laws/utils/_delays.py` and the draws of
`specs/_draw_spec.py`, which build one function per name inside a function. `_einstein.py` and `_pseudo_riemannian.py`
tell an unknown function apart from a library function by `isinstance(..., BuiltinFunction)`.

Of about 5700 methods in `src`, a rough count finds that between a quarter and a third begin with a word that is not a
verb, such as `member`, `lattice`, `views`, or `noise`.

## Goals / Non-Goals

**Goals:**

- Every module of `src/skcomplex` follows the rules on constants, helpers, method names, and symbolic functions, with
  the exceptions named here and in the constitution.
- Every Python file of `tests` and `docs` follows the rules on constants.
- Tests in `tests/test_sources.py` keep all of them true.
- Behaviour, public names, and performance do not change.

**Non-Goals:**

- The helper rule in `tests` and `docs`. Test functions and fixtures are module functions by design, and the gallery
  scripts and generators are scripts.
- `docs/generated`, which is written by the generators.
- Class-level constants. The constitution already makes them `ClassVar`.
- Lowercase module values that are part of the public API, such as `where`. They are function
  objects, not constants.
- Unknown functions. `function('a')` declares an unknown of an equation, the way `var('x')` declares an unknown
  variable, as in the cosmologies and the symbolic solver. It stays the one way to declare an unknown function.
- Any change of public names, signatures, or `__all__`.

## Decisions

### 1. Layout of the top of a module

A Python file of `src`, `tests`, or `docs` reads: docstring, licence header, imports, the constants block, the type
aliases, the other module statements, then the definitions in the constitution's order. A statement that must run
before an import, such as a path setup, belongs to the imports. The block holds one constant per statement run with
no blank line, no comment line, and no string between them, and no comment sits right above it. A blank line inside a
multi-line value counts as a gap too, so the few values that hold one are reformatted.

An attribute docstring under a constant is a comment in effect, so it goes. What it says moves into the module
docstring, or into the docstring of the function or class that reads the constant, where the reader needs it.

A constant annotation must not name a type alias of its own module, because Python 3.12 and 3.13 evaluate module
annotations when the line runs. Three constants of `systems/biology/systems_biology/_logical_networks.py` do that
today. They drop the annotation, and mypy checks the literal where it is passed. Putting the aliases first was
considered. It would make that case free, but it moves the aliases of more modules than it saves, and it contradicts
"constants right after the imports".

A constant whose value reads the state a statement sets, such as `SCRAPER` after `BUILD.mark_stale()` in
`docs/gallery_conf.py`, keeps its value and its order. The statement becomes part of the value or of the code that
reads the constant, whichever reads more plainly. The person applying the change checks each such case by hand.

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
definition it reads. The cases found are the reductions of `laws/utils/_reductions.py`, `MITTAG_LEFFLER`,
`POSITIVE_VALUES`, `STOP_GRADIENT`, `HELPERS_FINGERPRINT`, the four dialects and `DIALECTS` of
`specs/_dialect_spec.py`, `SYSTEM_CLASSES` of the catalog, and in systems the tracer and atom map tables of
`_carbon_13_mfa.py`, the landscapes of `_optimisers.py`, the predictors of `_el_farol.py`, the plants of `_koopman.py`,
the LAMMPS tables, the collisions of `_lattice_gases.py`, `SPIN_HALF`, and the links of `_yang_mills.py`.

After decision 4, no Sage hook is a module function. A plain function that a derived value reads, such as an entry of
a dispatch table or the predicate of a condition set, stays a module function beside the value. It cannot be a method
of a class that the value does not belong to, and the value must exist when the module loads.

Rejected: a cached class method that builds the value on first use. It changes when the value exists and its identity
across imports, and adds machinery.

### 4. Symbolic functions are `BuiltinFunction` subclasses

The user decided that a symbolic function the library defines is a `BuiltinFunction` subclass, as `where` and
`stop_gradient` already are, so there is one way to define one and its hooks are methods. Every call of Sage's
`function` or `formal_function` that defines a named library function becomes an instance of a private subclass.

- The subclass is named after what it is, `_<Name>Function`, as the constitution asks. The existing `_Where`,
  `_Either`, `_Both`, `_Mod`, `_Derivative`, `_Lag`, `_Delayed`, `_Residence`, and `_StopGradient` are renamed to the
  same pattern.
- Its `__init__` calls `super().__init__(name, nargs=..., latex_name=...)` with the settings the factory call held, as
  `_Where` does.
- Each hook becomes the method Sage reads: `print_func` becomes `_print_`, `eval_func` becomes `_eval_`, `evalf_func`
  becomes `_evalf_`, and `derivative_func` becomes `_derivative_`. The draws' derivative lambda becomes a
  `_derivative_` method that returns zero.
- Functions that share their hooks and differ only by name share one subclass that takes the name: the reductions over
  their kinds, the delays by order and kind, and the draws by label. Each such instance is built once per name and
  kept, so Sage registers each name once.
- The module value stays where it is read, as a derived value: `REDUCTIONS`, `NEIGHBORS`, `HYPEREDGES`, `COMPONENTS`,
  `SHORTEST_PATHS`, `SHIFT_MEMBER`, `NEIGHBOR`, `MITTAG_LEFFLER`, the Caputo derivative, and the operators of
  `_operators.py`, `_cells.py`, `_integrals.py`, `_contractions.py`, `_phase.py`, and `_local_operators.py`.
- Unknown functions keep `function(name)`, as the non-goals say.

A `BuiltinFunction` behaves unlike a factory function in three ways, and each conversion is tested for them. First,
on a non-symbolic argument it may call the method of the same name on the argument, so a name such as `components`
or `partial` is checked on numeric and on symbolic arguments. Second, `_einstein.py` and `_pseudo_riemannian.py` stop
treating library operators as unknown functions, which is the intent. Each is checked with its tests. Third, printing,
LaTeX, and the names the printers and the compiler read must stay the same, which the printer tests and the compile
tests confirm.

This group runs before the helper moves of `laws/utils` and `analysis`, so the hooks never become module functions
that move twice.

### 5. Which class owns a helper

The readers of a module function are the definitions that name it, followed through other module functions. A class
reader makes it a method.

- One class reads it: that class owns it.
- Several classes of the module read it: their closest common base defined in the same module owns it. Without one,
  the first reading class in module order owns it, and the others call `Owner._name(...)`. No base is added for this,
  since the constitution asks for the fewest classes.
- Classes of other modules import it: the class the defining module exists for owns it. When the defining module holds
  no class, the reading class whose concept the helpers serve owns them and the module merges into that class's module.
  `engine/compile/_complex.py` becomes static methods of `LowerComplexPass`, and `ObservableEvaluator` and
  `ReductionLowering` call it through `LowerComplexPass`. When no reader serves the concept, the helpers become
  public functions of the package `_utils` module, as the constitution asks for small general helpers. This is a move
  between modules of one package, never a package move.
- An owner must sit where every reader can import it without a cycle. When the natural owner would close a cycle,
  the helper goes to the package `_utils` module instead.

The kind of method follows what it reads. It is an instance method when it reads `self`, a `@staticmethod` when it
reads no instance state, and a `@classmethod` only when it needs `cls`, to read a class attribute a subclass overrides
or to build an instance. Inside the owner and its subclasses, instance methods call `self._name`. Static methods call
`Owner._name`, as `LayoutPass._declare_hidden` already does. Private methods come before public ones, in dependency
order, as the constitution asks. A moved function keeps its leading underscore and its one-line docstring.

A moved function passed as a callable, such as the diffusion coefficient of the reaction-diffusion systems, is passed
as `self._name` or `Owner._name`. Static methods pickle by their qualified name, so the parallel experiments keep
working.

### 6. Methods begin with a verb

The user decided that moved helpers with noun names take verb names now, as the constitution asks of methods. A name
that names the value it returns takes `get`, as most methods already do: `_noise` becomes `_get_noise`, `_richness`
becomes `_get_richness`. A name that names an action in another word takes that verb, such as `_build_`, `_read_`,
`_count_`, or `_draw_`. A name that already begins with a verb stays.

The check covers every method of `src`, not only the moved ones. A test cannot tell a moved method from an older one,
and the constitution's rule is about every method. So each package group also renames its older methods whose name
begins with a noun, such as `_member_*`, `_lattice_*`, or `_views`. The rough count above says this reaches between a
quarter and a third of the methods. The renames are scripted: a scratchpad script proposes a verb for each name,
the person applying the change reviews the list, and the script rewrites the definitions and their call sites.

The verb test reads the first word of each method name, without its underscores, and fails when the word is not in a
vocabulary of verbs. The vocabulary is a public constant of `tests/test_sources.py`, built once from the verbs the
code uses, and a new verb is added to it with the code that first uses it. These names are exempt: dunders, Sage
hooks written `_name_`, properties, and a name that a base class from another library defines, which the test finds
by importing the class.

### 7. Numba kernels

A function decorated with `njit`, `jit`, `vectorize`, `guvectorize`, or `cfunc` of `numba`, in any import form, stays
a module function. This is the exact rule, and it holds for public kernels such as `step_count` too. Kernels are
functions, not methods, so the verb rule does not reach them.

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

### 8. Helpers of public functions

The public API of `core/analysis`, `core/experiments`, `core/components/laws/utils`, and `engine/runtime/_simulate.py`
is functions, as the scikit-learn alignment decided, and some system modules re-export functions such as `life`. The
user decided that the private helpers only those functions read stay private module functions.

So a private module function that no class reads, in its module or in a module that imports it, stays. This covers the
shared readers of `analysis/_base.py` and the run helpers of `experiments/_runs.py`, which keep their names and their
imports within the package. A helper that a result class or any other class reads becomes a method of that class.

### 9. Public helpers that no package re-exports

A public function of a private module must be re-exported by its package, live in a `_utils` module, or be compiled.
`engine/_utils.py` already holds the engine's shared helpers and stays. `step_count` is compiled. `probe_laws` and its
private namespace helper move into `engine/_utils.py`, since passes and the replicate experiment read them, and
`compile/_probe.py` goes. The six system functions become private methods with verb names of the classes that read
them, under the owner rule. The examples in their docstrings move into the mirrored test modules, because a private
method carries only its summary line.

### 10. The automated checks

Five tests join `tests/test_sources.py`, next to the test of forbidden words. They read files with `ast` and
`tokenize`, and each failure names the path, the line, and the name.

- `test_module_constants_lead_their_module`, over `src`, `tests`, and `docs` without `docs/generated`: an upper-case
  module assignment must have no leading underscore. Every statement above it must be the docstring, an import, a
  statement before an import, or a constant, unless its value names a definition of the module or a derived value, in
  which case every statement between the last such name and it must be another derived value. No type alias may sit
  above a constant.
- `test_constant_blocks_hold_no_gap`, over the same files: no blank line, comment, or string between two constants of
  a block, none inside a constant's value, and no comment on the line above the block.
- `test_classes_own_the_functions_they_read`, over `src`: build the readers of every module function within its
  module and in every module that imports it. A private function fails when a class reads it, unless it is compiled or
  a derived value reads it. A public function of a private module fails when no `__all__` lists it, it is not compiled,
  and its module is not a `_utils` module.
- `test_methods_begin_with_a_verb`, over `src`: as decision 6 says.
- `test_symbolic_functions_are_builtin_subclasses`, over `src`: a call of Sage's `function` or `formal_function`, under
  any import alias, fails when it sits in a module-level assignment or passes a hook keyword ending in `_func` or a
  `latex_name`. A call with only a name, inside a function, declares an unknown and passes.

The tests are parametrised by the packages and folders they check, so each task group adds its own when it passes,
and the last group replaces the list with every folder. Nothing is skipped and nothing is allowlisted apart from the
verb vocabulary of decision 6. The scans take a few seconds, most of it the imports of the verb test.

Rejected: a ruff rule, since ruff has no plugin interface. A flake8 plugin, since it adds a second linter for one
check. The project already checks its sources in `tests/test_sources.py`.

### 11. How the refactor runs

Scripts in the session scratchpad do the mechanical work, never in the repository.

- The constants script hoists and renames constants and removes gaps, attribute docstrings, and comments in the block.
- The helpers script moves the text of each function into its owner's body with one more indent, adds `@staticmethod`
  where it reads no instance state, and rewrites call sites to `self._name` or `Owner._name`.
- The names script proposes a verb name for every method whose name begins with a noun, and rewrites the definition
  and every call site once the list is reviewed.
- The symbolic functions are converted by hand, since each has its own hooks.

A person reviews each diff, writes the docstring text that replaces an attribute docstring, and resolves the cases a
script flags: cycles, a clash of a moved or renamed method with an existing name, and functions read both by a class
and by a derived value.

After each package: `ruff format`, `ruff check`, `mypy`, the mirrored tests, and the doctests of the package pass.
When several agents run tests at once, each sets the Numba, OpenMP, and BLAS thread counts to 1.

## Risks / Trade-offs

- [The earlier changes move many of these modules] → This change lands last and the first task measures again.
- [The verb renames reach many existing methods] → They are scripted and reviewed per package. A renamed hook that a
  subclass in another package overrides is renamed in the same commit, which the names script finds by searching the
  whole tree for each old name.
- [A `BuiltinFunction` evaluates differently from a factory function] → Each conversion is tested on numeric and
  symbolic arguments, and the printer, compile, Einstein, and metric tests run after it, as decision 4 says.
- [A moved owner closes an import cycle] → The owner goes to the lower module or to the package `_utils` module, as
  decision 5 says, and `lint-imports` confirms it.
- [A moved or renamed method shadows an inherited method] → The scripts report every clash, and the method takes a
  more specific name.
- [Renamed constants invalidate the Numba cache once] → The fingerprint hashes the kernel modules, so the first run
  after the change recompiles. This is expected and harmless.
- [Callables passed to parallel workers no longer pickle] → Static methods pickle by qualified name. The experiment
  tests that run in parallel confirm it.
- [Hoisting a constant in an example changes the story it tells] → The example's text cells still introduce each
  value where it is used, and the gallery builds as before.
- [Attribute docstrings carried knowledge] → Their text moves to the docstring that a reader of the constant reads.
- [A large diff is hard to review] → One commit per package, each with its checks passing.

## Migration Plan

The work lands package by package on `development`, after the four earlier changes. Each commit passes the full gate,
so any one can be reverted alone. No public name changes, so users need no migration.

## Decisions by the user

The user answered the four open questions of the first draft on 2026-11-09.

1. Helpers that only public functions read, in analysis, experiments, law utilities, and `simulate`, stay private
   module functions. Decision 8 applies it.
2. Every symbolic function the library defines becomes a `BuiltinFunction` subclass, as `where` and `stop_gradient`
   already are, so its hooks are methods and there is one way to define a symbolic function. Decision 4 applies it, as
   its own task group.
3. Moved helpers with noun names take verb names now, following the constitution, and the check covers it where
   feasible. Decision 6 applies it. A test cannot tell moved methods from older ones, so the check and the renames
   reach every method of `src`. If the user wants the renames limited to moved helpers, the verb test is dropped and
   the names script runs only on the moved methods.
4. The constants rule applies to `src`, `tests`, and `docs`, the examples and the generators included, and the same
   test checks it. Decisions 1 and 10 apply it.
