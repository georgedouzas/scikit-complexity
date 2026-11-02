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
- choice: `'uniform'`, `'score'` with a score callable, or `'nearest'`;
- fallback: `None` or `'nearest'`;
- capacity: a number or a state name.

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
