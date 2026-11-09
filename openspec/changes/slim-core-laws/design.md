## Context

See proposal.md for the motivation. The paths below are those after `add-system-package`, which moves
`components/system.py` and `mass_action.py` into `components/systems/`. Nothing in this change touches those modules.

The audit of the laws (`audit-laws.md`) was checked against the code. These of its claims are wrong, and the plan
follows the code:

- `circular_mean` has no user in `src/skcomplex/systems`. The flocking, collision, and leadership modules it named
  compute their headings without it. Moving it into private methods
  of those modules would add code that nothing calls. It stays, as a named reduction of directional statistics.
- `lookup` has no user in `chemistry/kinetics/_reactors`. Its system users are `social/organizations/_traps`,
  `economics/resources/_food`, `_plantations`, `_extraction`, `economics/markets/_commodities`,
  `physics/climate/_pollution`, `engineering/processes/_systems`, and `biology/ecology/_factories`, plus the system
  dynamics translator.
- `delay1` has no user outside its tests. `delay3` and `smooth` have one system user,
  `engineering/processes/_systems`, and the translator writes all three.
- `None` is not accepted as a sink. `TransferLaw._check_holders` rejects any holder that is not a symbol or
  `Source()` and `Sink()`. In `TransferInteraction`, `into=None` means the same state as the source leg, not the
  environment. So `Source()` and `Sink()` are already the only way to name the environment, and nothing changes.
- The draw functions have one public path today, `laws.utils`. `skcomplex.core.specs` does not re-export them. The
  fault is that `laws.utils` re-exports names that live in `specs/_draw_spec.py`, against the rule that a name is
  re-exported once, where it lives.
- `dagger` and `operator_product` are not compiler helpers. They build operators in laws, as the module docstring of
  `_local_operators` and the user guide page `quantum_systems.md` show. They stay public. `has_operator` is the
  compiler helper of that module.
- `EncounterLaw` has the law tag `rate`, not `encounter`, and declares no state domain. The `kinetic-models` delta
  states what the code does.
- `laws/utils/_tables.py` holds only a docstring. It is removed.

The constitution treats a package and its subpackages as one package for imports, so `skcomplex.core.engine` may
import a name from a private module of `skcomplex.core.components.laws.utils`. `engine/problems/_operator_lowering.py`
already imports `LOCAL_OPERATOR` this way. A helper is private when its package does not re-export it.

## Goals / Non-Goals

**Goals:**

- One law class or one utility for each operation, with the parametrised copies folded in.
- `laws.utils` re-exports only what a law is written with.
- Every removed meaningful piece of code, the checks, docstrings, and examples, moves into the surviving part.

**Non-Goals:**

- The names, arguments, and docstrings of interactions. `slim-core-interactions` owns them. This change edits only
  the lines of interactions that construct a removed law or import a privatised helper.
- `periodic_guard`, which the interactions audit proposes for `ImpulseInteraction`. If `slim-core-interactions`
  needs it, it adds it to `laws.utils` there.
- `_SubsystemSpec`, `HolonomyScoreSpec`, and `RegionScoreSpec`. They are specs of components and interactions.
- The capability tags. Every merged law keeps the law tag of its survivor, so the inventory and the catalog tags do
  not change.

## Decisions

### Leave and merge table

| From | To | Realizations per field | Core dependents |
| --- | --- | --- | --- |
| `FractionalNoiseLaw(target, value, order, *, hurst, convention, label)` | `NoiseLaw(..., hurst=None)` | None. Only `RateInteraction` and `RelaxationInteraction` emit it | `interactions/_rate`, `_relaxation`, `specs/_interaction_requirement_spec`, `engine/compile/passes/_check_fractional_noise`, `_fractional_problem`, `_pass_compiler` (noise census), `solvers/_fractional_euler_maruyama`, `_euler_maruyama`, `_heun`, `_cosimulation`, `backends/_numpy`, `_numba`, `core/types.NoiseKind` |
| `EmissionLaw(target, source, matrix, *, label)` | `TransitionLaw(target, matrix, *, source=None, label)` | None. Only `TransitionInteraction` emits it | `interactions/_transition`, `engine/compile/passes/_markov`, `solvers/_maps`, `_cosimulation`, `backends/_numpy`, `_numba`, `experiments/_replicate` |
| `EncounterLaw(candidate, field, rate, transition, test, *, label)` | `BilinearLaw(target, first, second, rate, outcomes=None, *, conserved=False, label)` | None. Only `EncounterInteraction` emits it | `interactions/_encounter`, `engine/types.KineticLaw`, `compile/_laws`, `compile/_pass_compiler`, `compile/passes/_compile_encounters`, `solvers/_kinetics`, `_rk4`, `_scipy_ode`, `_strang_splitting`, `backends/_numpy`, `_numba` |
| `DriveLaw` | `ArrayUpdateLaw` (rename, module `_array_update.py`) | None. Systems use it directly: `physics/common/spins/_spin_sampling`, `biology/systems_biology/_fuzzy_oil_drop` | `laws/_base` docstring, `interactions/_sampling`, `_cascade`, `compile/_forms.LATTICE_LAWS`, `compile/_laws`, `compile/passes/_lattices`, `solvers/_lattice`, `backends/_numpy`, `_numba` |
| `RateLaw(..., cross_section=...)` and `collision(f)` | `RateLaw` without `cross_section`, `collision_integral(density, kernel)` | None. `physics/fluids/_kinetic_equations` reaches it through `EncounterInteraction` | `interactions/_encounter._get_laws`, `compile/_field_lowering`, `compile/_pass_compiler._collision_issue`, `problems/_collision_kernel` |
| `lookup(value, points)` | `series(x, points, values, kind='linear')` | None. The eight system modules listed in Context call `series` | `engine/translators/_expressions`, `_module_writer` |
| `delay1`, `delay3`, `smooth` | `distributed_delay(value, time, order=1, *, conserved=True)` | None. `engineering/processes/_systems` calls `distributed_delay` | `engine/translators/_expressions` |
| `inverse_laplacian(x)` | `inverse_helmholtz(x, screening=0, boundary=None)` | None. `physics/active_matter/_nematics`, `physics/fluids/_kinetic_equations`, `physics/fluids/_flows` call `inverse_helmholtz` | `laws/utils/_tensors.leray_projection` |
| `random_pairing` | private method `MatchLaw._get_random_pairs` | None | `laws/_match` |
| `uniform`, `normal`, `bernoulli`, `categorical`, `draw`, `draws_in`, `DRAWS` in `specs/_draw_spec.py` | `laws/utils/_draws.py`, `DrawSpec` stays in specs | None | `laws/utils/_reductions`, `_conditions`, `compile/passes/_renewal_problem`, every engine module that imports `draws_in` |
| Public `draws_in`, `reductions_in`, `neighborhoods_in`, `hyperedges_in`, `contractions_in`, `cell_operations_in`, `delay_parts`, `has_operator`, `distribute_operators`, `coordinates_of`, `COORDINATE_PREFIX`, `INVERSION_BOUNDARIES`, `MAX_NEIGHBOR_HYPEREDGE_SIZE`, `NEIGHBOR_REDUCTIONS` | Private: dropped from `laws/utils/__init__`, imported from their private module | None | `laws/_generator`, `_dissipator`, `interactions/_grouping`, `_gradient`, `analysis/_loops`, the compile passes, `problems/_operator_lowering` |
| `laws/utils/_tables.py` (empty) | Removed | None | None |

### Kept

`ExtremalLaw` stays, because it names a math operation and the user listed it among single-field mechanisms that stay.
`Source` and `Sink` stay as the only names of the environment. `circular_mean`, `dagger`, `operator_product`, `step`,
`ramp`, `pulse`, `pulse_train`, `series`, `clip`, `sign`, `hill`, and `logistic` stay public. Every other law keeps
its signature.

### No realizations

The user decided that a field gets a subclass only when it has its own term, and that pure math conveniences and laws
only core interactions emit get none. The audit proposed `FractionalBrownianNoiseLaw`, `HiddenMarkovEmissionLaw`,
`BinaryCollisionLaw`, `KineticExchangeLaw`, `WealthExchangeLaw`, `ImmuneEncounterLaw`, `table_function`, `mean_heading`,
`material_delay1`, `poisson_potential`, and others. No system constructs the three laws directly, and `table function`,
`material delay`, and `Poisson potential` are system dynamics and math terms, not the terms of one field. So no copies
are made, and the systems call the core part. The docstrings and examples of the removed parts move into the docstring
of the survivor, which is where their meaning now lives.

### `NoiseLaw` with a Hurst index

`NoiseLaw` takes `hurst=None` after `conserved`. `_prepare` keeps the range check of `FractionalNoiseLaw` and adds two
checks: a Hurst index with a process, and a Hurst index on a conserved noise, each raise a `ValidationError`. The
fractional stepper is chosen by a new member `NoiseKind.FRACTIONAL`, which the noise census of `PassCompiler` records
for a law whose `hurst` is not `None` after `CheckFractionalNoisePass` has turned a Hurst index of 1/2 into a plain
noise. `FractionalEulerMaruyamaSolver` supports `{DIAGONAL, FRACTIONAL}`. `EulerMaruyamaSolver`, `HeunSolver`, and
`CoSimulationSolver` support every kind but `FRACTIONAL`, and the fix text of the census names the fractional
stepper, so the "Wrong stepper" scenario keeps its message.

The alternative was a per-instance law requirement. `SupportSpec.laws` holds classes, and a solver would then need a
second check of instances. The noise kind already describes what a stepper can draw, so it is the one place.

### `TransitionLaw` with a source

`TransitionLaw` takes `source=None` as a keyword. With a source, `_prepare` raises for a function matrix, with the
message of `EmissionLaw`, and checks a rectangular matrix. Without one it behaves as today. `MarkovPass` tests
`law.source is not None` where it tested the class.

### `BilinearLaw` with a conserved loss

`BilinearLaw` takes `conserved=False` as a keyword. An `EncounterLaw(candidate, field, rate, transition, test)` is
`BilinearLaw(test or candidate, candidate, field, rate, transition, conserved=True)`. A conserved law needs an outcome
table, and `_prepare` raises without one, because the identity outcome makes the gain and the loss cancel. `second`
may be `None`, which is the linear redistribution `EncounterLaw` allowed. `CompileEncountersPass` reads the kind
`'transition'` or `'bilinear'` from `law.conserved` and `linear` from `law.second is None`. `engine/types.KineticLaw`
and `_KINETIC_LAWS` are removed, since one class remains. `StrangSplittingSolver` lists `BilinearLaw`, and its inner
collision solver, RK4 by default, already advances both kinds.

### `ArrayUpdateLaw`

The rename follows the operation, a new array of a state from a function of the old one. The arguments do not change.
The module becomes `laws/_array_update.py` and the test module `tests/core/components/laws/test_array_update.py`.

### `collision_integral(density, kernel)`

A Python callable cannot sit inside a Sage expression. So `collision_integral` calls the kernel once with two private
real symbols, the relative speed and the deflection cosine, and returns the unevaluated call
`collision_integral(density, B(speed, cosine))`. The Maxwell kernel `lambda speed, cosine: 1 / (2 * pi)` gives a
number, and a kernel that uses Sage functions gives an expression. `FieldLowering._collision` reads the kernel from the
second operand and turns it into a NumPy function of the two arrays for `CollisionKernel`. It raises for a kernel that
reads a state or a parameter, because the weights are assembled once. `RateLaw.cross_section` and
`FieldLowering.cross_section_` are removed, and `PassCompiler._collision_issue` looks for the call in the law value.
`PHASE_OPERATOR_NAMES` names `collision_integral`.

The alternative was to keep the kernel on the law. The user asked to move it off `RateLaw`, and a kernel in the
expression lets any law that reads the integral run without a special argument.

`EncounterInteraction._get_laws` is the one emit site. This change edits that line to emit
`RateLaw(candidate, factor * collision_integral(candidate, self.cross_section))`. The argument name `cross_section` of
`EncounterInteraction` and its docstring are the business of `slim-core-interactions`.

### `series` and `distributed_delay`

`series(x, points, values, kind='linear')` takes any value as `x`. The checks and the binary search do not change.
The tables of the system modules become pairs of tuples of points and values. The translator writes
`series(x, points, values)` for a lookup, `distributed_delay(x, T, n)` for `DELAY1`, `DELAY3`, and `DELAY N`, and
`distributed_delay(x, T, n, conserved=False)` for `SMOOTH` and `SMOOTH N`.

### `inverse_helmholtz`

`screening` gets the default `0`, so the Poisson solution is `inverse_helmholtz(x)`. The call form does not change,
and `inverse_laplacian(x)` already returned `inverse_helmholtz(x, 0)`.

### Private helpers

The helpers stay in their private modules, and `laws/utils/__init__` stops re-exporting them. Their users import
them from the private module. `random_pairing` is used by `MatchLaw` alone, so it becomes the private method
`MatchLaw._get_random_pairs`, by the rule that helpers a class uses are its private methods. `coordinates_of` stays a
module function of `_phase.py`, because `_moments.py` and `_integrals.py` use it too. The constants keep their
`UPPER_CASE` names without an underscore, as the constitution asks.

### Draw functions

`uniform`, `normal`, `bernoulli`, `categorical`, `draw`, `draws_in`, the registry `DRAWS`, and `_call` move to
`laws/utils/_draws.py`. `DrawSpec` and `RELAXED_KINDS` stay in `specs/_draw_spec.py`. `laws.utils` is the lowest layer
that names the functions as law vocabulary, and `tests/core/components/laws/utils/test_draws.py` already sits there,
so the tests mirror the source again.

## Risks / Trade-offs

- [`slim-core-interactions` edits the same interaction modules] → This change touches only the constructor lines and
  imports listed in the table. Whichever change lands second rebases those lines. If the interactions change removes
  `RelaxationInteraction` first, its `FractionalNoiseLaw` site goes with it.
- [`general-mechanisms` "Bilinear laws are one law" may also be touched by the interactions change] → The law-level
  requirement belongs here. The interactions change modifies only the interaction requirements, such as "Encounters
  take a collision integral".
- [A kernel written with NumPy functions fails on Sage symbols] → The docstring of `collision_integral` says the
  kernel returns an expression. The only kernels in the tree are constants.
- [Systems that called `lookup` with a list of points now keep two tuples] → The tuples sit next to each other as
  module constants, so a table reads as before.
- [`tests/core/components/elements/test_density.py` names `EncounterLaw`] → This change edits only the law there. The
  elements change owns the element names.

## Migration Plan

The change is breaking with no shim. A user replaces `FractionalNoiseLaw(x, a, hurst=h)` with
`NoiseLaw(x, a, hurst=h)`, `EmissionLaw(y, x, M)` with `TransitionLaw(y, M, source=x)`,
`EncounterLaw(f, g, r, A)` with `BilinearLaw(f, f, g, r, A, conserved=True)`, with the test density as the
target when there is one, `DriveLaw` with
`ArrayUpdateLaw`, `RateLaw(f, collision(f), cross_section=B)` with `RateLaw(f, collision_integral(f, B))`,
`lookup(x, points)` with `series(x, xs, ys)`, `delay3(x, T)` with `distributed_delay(x, T, 3)`, `smooth(x, T)` with
`distributed_delay(x, T, conserved=False)`, and `inverse_laplacian(x)` with `inverse_helmholtz(x)`. The changelog of
the release lists these under breaking changes.
