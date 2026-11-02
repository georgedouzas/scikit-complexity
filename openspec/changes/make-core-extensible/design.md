# Design

## Context

See proposal.md for the motivation. The audit tables are the starting point:
`interactions_elements_table.md` (37 interactions, 8 elements, the views) and `laws_engine_table.md` (laws, specs,
engine paths), kept in the session scratchpad and summarised in this design.

The relevant state of the code:
- **Roles:** `BaseInteraction._get_role(role)` returns the role unchanged. The agent forms and the default
  `TransferInteraction._get_holders` call it. About 30 symbolic `_get_laws` read `namespace.target`, `.source`,
  `.candidate`, `.field`, `.first`, `.second` and `.member` directly.
- **Parameters:** agent forms read `model.parameter('<name>')` and symbolic forms read `namespace.<name>` with literal
  names. `EvolutionInteraction._value` already reads a parameter when the class declares it and the plain attribute
  otherwise.
- **Variants:** module constants (`PAIRINGS`, `SCHEDULES`, `STEPS`, `RULES`, `GROUPS`, `SELECTIONS`, `REPLACEMENTS`,
  `PLACEMENTS`, `MODES`, `ACTIONS`) checked in `_check` and dispatched with `if`/`elif` to private methods that a
  subclass could override, but `super()._check()` rejects any new value first.
- **Partners:** three mechanisms: seven transfer pairings with a group `fallback`, eight matching rule specs (greedy with
  `candidates`, `score`, `capacity`), and four groupings. `MatchLaw` with the `Pairing` callable alias exists but no
  interaction uses it.
- **Callables:** rate, noise, update, boundary laws and the jump rate take `f(t, values)` or `CallSpec(f, args)`,
  lowered through `KernelsPass._outputs_of` into `LawKernel._call`, in Python on Numba unless the function is jitted.
  Crossings, fluxes, transition matrices, jump changes, renewal samplers, reductions, and observables of non-agent
  systems take symbols only. Callables read global dotted paths and receive no random stream.
- **States:** `StateSpec(domain=None)` lets an agent hold a tuple, but snapshots reshape it, compositions call
  `float`, histories drop it, and analyses fail on it.

## Goals / Non-Goals

**Goals:**
- One mechanism per kind of hook, on `BaseInteraction` or in `skcomplex.specs`, used by every general class.
- Every default keeps today's behaviour and trajectories, except where today's result is a silent wrong result.

**Non-Goals:**
- Compiling arbitrary callables with Numba. A callable runs in Python unless it is jitted, as today.
- Symbolic analysis, exact Jacobians, or conservation proofs of callable laws. They stay symbolic-only, with the errors
  they raise today.
- Object-valued states for symbolic (non-agent) systems, which the NumPy and Numba state vectors cannot hold.

## Decisions

### Three hooks on `BaseInteraction`
`_get_role(role)` exists. Add `_get_member(namespace, role)`, returning `getattr(namespace, self._get_role(role))`, and
use it in every symbolic `_get_laws` instead of `namespace.<role>`. Add `_get_parameter(name)`, returning the name by
default, used by every `model.parameter(...)` and `namespace.<parameter>` read. A parameter the class does not declare
falls back to the plain attribute, as `EvolutionInteraction._value` does, so a subclass fixes a value by setting it.

Alternative: a mapping argument `roles={'target': 'stock'}` on every class. Rejected: subclasses already override
methods for the same purpose, and an argument would be a second way.

### Open variants

Decided on 2026-11-02: the choice and imitation rule, grouping `groups`, evolution selection and replacement, topology
action, and cascade, coupling, and map modes accept their named values or a spec a user defines by subclassing the
interaction's private dispatch method; only pairings, steps, and placements, which have a natural signature, also take
callables.


Each variant argument accepts its named values, a spec of the matching family in `skcomplex.specs`, or a callable. The
check accepts a spec or a callable and keeps validating the rest. Dispatch goes to the spec's private method on the
interaction for specs it knows, and calls the callable otherwise. Named values stay as shorthands for today's specs.

Alternative: registries of variants. Rejected: global mutable state and machinery the project avoids.

### One partner-selection spec
`PartnerSpec(candidates, choice, fallback, capacity)` in `skcomplex.specs`:
- candidates: `'all'`, `'neighbors'`, `'cell'`, a radius, or a state name holding a per-agent radius;
- choice: `'uniform'`, `'score'` with a `score` callable, `'nearest'`, or `'all'` for every candidate, which the named
  pairings `'neighbors'` and `'cell'` need;
- fallback: `None` or `'nearest'`;
- capacity: a number or a state name, the largest number of pairs of a member in one step. With a capacity the agents act
  in random order, since the first to act take the room, and it counts both roles within one group.

The named pairings `'random'`, `'pairs'`, `'neighbors'`, `'neighbor-pairs'`, and `'cell'` are shorthands of specs and give
identical draws. `'members'` and `'site'` select no partner, so they stay as they are. In groupings `'pairs'` is a
shorthand, and the groups of a spec or a function hold every member with its partners. A `'neighborhood'` group keeps its
own path because it also holds a member with no neighbours. A matching takes a spec or a function as its rule, and matches
its pairs at the midpoint of both limits. `GreedyMatchingSpec` keeps its own candidates, because its feasibility reads the
limits and quantities of the offers, which a spec does not know. A capacity does not run under the transfer schedule
`'member'`, since every actor draws alone, and it raises an error.

A pairing callable replaces the spec. Transfers, matching (as the greedy rule's candidates and choice), and groupings
accept it. Radius candidates use `RadiusNeighborhood` and `NearestNeighborhood` from `core.spaces`, so they work in
continuous space. `MatchLaw` becomes the lowering the three interactions share, so the unused law gains its users.
Transfer legs may credit both ends from one draw: a settlement callable receives both partners and the drawn values.

This reverses the third-round decision of `complete-general-model-rewrites` to keep the obesity meal in its model: the
audit showed "within reach, else the nearest" is a general rule (facility choice, foraging, service assignment).

### Callables on every numeric path
- Crossing guards and resets are evaluated numerically by `EventKernel`, so a callable passes straight through.
- `FluxLaw(speed=callable)` gives the Rusanov wave speed; Godunov stays symbolic.
- Transition matrices, jump changes, renewal samplers `f(rng, **args)`, reduction functions, and
  `ObservableSpec(function=f(t, values))` take callables on NumPy, with a Python fallback on Numba and its warning.

Callables gain three options on `CallSpec`: `relative=True` passes values keyed by the component's own names,
`random=True` passes the simulation's stream, `vectorized=True` calls once with arrays over a group. The plain
`f(t, values)` form stays as it is.

### Object states for agents
A state declared with `domain=None` or an object domain holds any hashable value. Mesa snapshots store 1-D object arrays,
`composition` counts by hashable value, `history` keeps any value, and `analysis._base` refuses object states by name.
Snapshot fields get explicit roles (position, velocity, heading) in the snapshot spec instead of being guessed from
names.

### Errors instead of silent results
The fixes listed in the proposal are checks in `_check` or in the pass that would otherwise run silently, each raising
a named error from `skcomplex.errors` whose message names the operation, the reason, and the hook.

### Restrictions the core keeps

- **Growing hypergraphs.** A hyperedge interaction on hyperedges that change during a run returns rates, noises, and
  jumps of its members, or updates of the hyperedge member. The compiler switches off the free rows of the group on
  hyperedges by multiplying each value by the activity of its hyperedge, which only a symbolic value allows, so a
  callable value raises an error. Lifting it needs the activity passed to callable laws as a value they read.
- **Grid variants in continuous space.** Cell, neighbour, and site pairings, neighbourhood groups, neighbour steps and
  placements, and neighbour candidates read the cells or nodes of a grid or network, which a continuous space does not
  have. They raise a `SpaceError` at `prepare` naming the variant that runs there. The radius candidates of the
  partner spec lift it for pairings and candidates.
- **Update phases.** Update laws change all members at once in their phase, so a phase of update laws takes only a
  period and an offset. An activation or a repeat raises an error, and `UpdateLaw(order=...)` orders members instead.
- **Stratonovich and anti-Itô noise.** The drift correction needs the derivative of the amplitude, which the compiler
  takes symbolically, so a callable amplitude of either convention raises an error. A `derivative=` argument would lift
  it.
- **Field boundaries and discretizations.** A field takes the boundaries `'periodic'`, `'zero-flux'`, and `'fixed'`
  and the discretizations `'finite-difference'`, `'spectral'`, `'finite-volume'`, and `'finite-element'`, because each
  needs its own lowering of the operators. A new one is added as a lowering in the engine, not by the user.
- **Group constructors.** A group builds one instance of its element type from the label alone and sets its `values`
  on it, so every other argument of the element constructor has a default. Per-member values go in `values`.
- **Event ties.** The crossings of one step apply earliest first. Crossings within the minimum separation of the
  earliest, a relative `1e-10`, are one tie: all of them fire at that time, by declaration order and then member index,
  and each reset reads the state the one before it left. The reason is that a reset is a function of one state vector,
  so resets applied in turn need no rule for two resets that write one state. An option on the event locator that
  evaluates every reset on the state before the tie, with a rule for conflicting writes, would treat them as
  simultaneous. Until 2026-11-02 the code fired only the first of a tie, because the guards of the others had already
  changed sign when the run restarted; `EventLocator.apply` now fires all of them.
- **Draw registry.** `DRAWS` in `skcomplex.specs._draw_spec` is a module-level mapping from the function name of a draw,
  such as `uniform_turn`, to its `DrawSpec`, written by every call of `uniform`, `normal`, `bernoulli`, `categorical`,
  and `draw`. The reason is that a Sage symbolic function holds only a name and arguments, so `draws_in` recovers the
  kind, the name, and the keywords of a draw from the name of its function. Entries never leave and no run writes them,
  so they do not change between runs, but two limits follow. An expression unpickled in a process that never called the
  draw is not recognised as a draw, and `uniform('a')` and `draw('a', 'uniform')` share the function name `uniform_a`,
  so the later call decides the kind of both. A Sage function class that holds the spec on the operator would lift both.

## Risks / Trade-offs

- [About 30 symbolic sites change at once] → Mechanical edits behind a default that returns today's names, with the
  full suite and a role-renaming test per general class.
- [Open variants weaken validation] → Specs are frozen and validated themselves; callables are called with a documented
  signature and their return is checked.
- [Partner unification touches three interactions that models depend on] → Named pairings stay as shorthands with
  identical draws; each model using them is compared over 20 seeds.
- [Object states slow snapshots] → Only states declared as objects take the object path.
- [Errors replace silent results some models relied on] → Each such model is reported with its old and new behaviour.

## Migration Plan

Additive for users. Work on `development` in groups: hooks first, then partners, then callables, then object states and
errors, merging each group with the full suite green.
