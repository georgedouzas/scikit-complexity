# Proposal

## Why

The library aims to be scikit-learn-like: users extend general core classes for any field. An audit of the core on
2026-11-01 found the opposite in places. Roles renamed by a subclass work only in agent forms, so a symbolic subclass
fails with a raw `AttributeError`. Parameters and variants are fixed names. Several law kinds take no callable even
though the `system-modeling` spec promises one. Agent states must be numeric. And some combinations give wrong results
silently, such as a callable Stratonovich amplitude that runs as Itô. This change removes those restrictions, or names
them in a clear error and a recorded reason.

## What Changes

- **Hooks for every fixed decision.**
  - Roles: a subclass that renames its roles works in the symbolic forms as well as the agent forms.
  - Parameters: a subclass names its own parameters, or fixes a value without declaring one.
  - Variants: pairings, rules, steps, groupings, selections, placements and actions accept a spec or a callable besides
    today's names, so a subclass adds a variant without bypassing its parent's checks.
  - Schedules and ties: the agent forms that are always synchronous gain a `schedule`, and the classes that break ties
    gain a `ties` argument.
  - The agent and model view protocols declare every member the views offer.
- **One partner-selection spec.** Candidates within a radius read from a state, a choice among them, a fallback to the
  nearest partner, a capacity, or a pairing callable. It reuses `MatchLaw`, works in continuous space through the radius
  and nearest-k neighbourhoods, and is accepted by transfers, matching and groupings. The obesity meal is built on it
  and leaves its model.
- **A callable fallback on every numeric path.** Crossing guards and resets, a flux wave speed, transition matrices,
  jump changes, renewal samplers, reductions, and observables of non-agent systems take callables. Every callable law can
  read values keyed by its component, receive the simulation's random stream, and run once over a whole group. Law kinds
  that need symbols by nature (equations, quantum generators and dissipators, Godunov fluxes) say so in their error.
- **Object values for agent states.** Agent states may hold hashable objects. Compositions count them, snapshots keep
  them, histories record them, and analyses refuse them by name.
- **No silent wrong results.**
  - A callable Stratonovich amplitude raises an error or takes a supplied derivative.
  - A phase activation other than the default on update laws is refused.
  - Transitions over a cell honour collection and callable sources.
  - Matching rejects a missing or misspelled state instead of defaulting it.
  - Agent imitation honours `layer`.
  - Grid-only operations in continuous space raise an error naming the space they need.
  - Analyses no longer guess snapshot fields from names.
- **Every remaining restriction recorded** in a spec or design with its reason, including the event tie order, the draw
  registry, and the growing-hypergraph limits.

## Capabilities

### New Capabilities

- `core-extensibility`: how users extend the general core classes: hooks for roles, parameters, variants, schedules,
  and ties; callable laws with component-relative keys, the random stream, and batch evaluation; clear errors for every
  limit; and recorded reasons for every restriction.

### Modified Capabilities

- `system-modeling`: "Laws describe how state changes" now lists which law kinds take callables and which need symbols,
  matching the code.
- `general-mechanisms`: adds one partner-selection spec shared by transfers, matching, and groupings.
- `agent-based-modeling`: agent states take hashable object values.

## Impact

- **Code:** `skcomplex.core.interactions` (nearly every class, through shared hooks on `BaseInteraction`),
  `skcomplex.core.laws` and `laws.utils`, `skcomplex.specs` (the partner-selection spec, callable fields),
  `core/types.py` protocols, `engine/compile/passes` (noise, functions, events, flux, observables),
  `engine/solvers/_mesa.py` (object states, views), and `analysis/_base.py` (snapshot roles).
- **API:** additive. Every new argument defaults to today's behaviour. Errors replace today's silent wrong results,
  which is intended.
- **Models:** the obesity meal leaves its model; a model whose result relied on a silent wrong result changes and is
  reported.
- **Dependencies:** none.
- **Order:** after `complete-general-model-rewrites` and `add-higher-order-interactions` are archived. Insert it into the
  approved waves where the maintainer decides.
