# Proposal

## Why

The bases name their families but neither define them nor speak a common language.
- **The bases do not define what their children do.** `BaseSpace` declares no `size`, although five of its six spaces
  have one. `BaseProblem` defines none of the twenty methods the runtime calls on problems, so the runtime reaches
  for them with `getattr`.
- **Every family has its own verbs.** Solvers `initialize` and `advance`, backends `lower`, visuals `views` and
  `log`, observers react `on_start` and `on_frame`, compilers `claims` and `compile`, designs return `rows`, and
  surrogates are built by `fit_*` functions and `forecast`. A modeller learns a new vocabulary for every family.
- **Constructors do work.** Many `__init__` methods validate or derive attributes, so building an object can fail,
  and its state mixes what the user gave with what was computed.
- **Behaviour hides in magic methods.** There are about 90 hand-written ones:
  - 41 `__post_init__`;
  - 31 `__repr__`;
  - `__call__`, `__getattr__`, `__eq__`, `__hash__`, `__reduce__`, and `__new__`.

## What Changes

- **One vocabulary for every configurable base.** Each base on `BaseEstimator` speaks three verbs, with
  scikit-learn's `get_params` and `set_params`:
  - **`check()`** validates the stored arguments, raises a named error, and returns `self`.
  - **`prepare(context)`** prepares the object against what it works on, derives every attribute that ends in `_`, and
    returns `self`.
  - **`run(input)`** is the one action of the family, which uses the prepared state. `run(..., final=True)` also
    releases what `prepare` acquired.

  Each verb is written once on its base and calls a private hook, `_check`, `_prepare`, or `_run`, that a child
  overrides, so a base answers every verb, trivially where a family has nothing to do.
  - Families stop having verbs of their own: `initialize`, `advance`, `lower`, `views`, `log`, `on_start`,
    `on_frame`, `on_stop`, `claims`, `compile`, `forecast`, and `rows` become `prepare` and `run`.
  - Facts of a family become prepared attributes, such as `space.size_`, `space.positions_`, and `solver.events_`.
- **Trivial constructors.** `__init__` stores every argument unmodified as `self.x = x` and does nothing else. Every
  other attribute ends in `_` and is set by `prepare` or `run`. The checks in today's constructors move to `check()`.
- **Values keep data, not verbs.** Laws, domains, problems, terms, and compiled records stay frozen dataclasses
  without hand-written methods. Their only shared method is `check()`, and their bases define the fields every child
  has, such as `targets` for laws.
- **No hand-written magic methods** apart from `__init__` and the arithmetic of the quantum operators:
  - Collections are read with named methods and properties: `trajectory.values('a.n')` replaces `trajectory['a.n']`,
    `layout.size` replaces `len(layout)`, and `layout.entries` replaces iterating the layout.
  - The symbol namespace of laws keeps `s.x`, with every symbol stored as a plain attribute and no `__getattr__`.
- **Terms become value records** compared with `==`.
- **Surrogates become estimators:** `SINDySurrogate(...).prepare(series).run(50)` replaces `fit_sindy(...)` and
  `forecast`.
- **BREAKING:** every renamed verb, the prepared attributes that replace methods, construction-time validation, the
  surrogate functions, the design operators, the callable objects, and the repr of the classes that printed
  themselves.

## Capabilities

### New Capabilities
- `base-interfaces`: the shared verbs of the configurable bases, trivial constructors and prepared attributes, the
  values and their checks, and which magic methods the code allows.

### Modified Capabilities

## Impact

- **Bases:** every configurable base and its children:
  - components, interactions, spaces, times;
  - solvers, backends, system compilers, passes;
  - visuals, observers, clocks, designs, surrogates.

  Every value base and its children: laws, domains, problems.
- **Callers:**
  - the runtime, the compiler, and the visuals, which call the new verbs and read prepared attributes;
  - every model and example that calls a renamed method;
  - every `is` comparison of a term.
- **Depends on:** `gather-base-classes`. The change `fix-class-relations` reworks the quantity classes, so this change
  leaves `Auxiliary`, `Invariant`, `Objective`, and `Wiener` to it.
