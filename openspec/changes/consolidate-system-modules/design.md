## Context

The subfield packages of `skcomplex.systems` split each system into a module that registers it and several one-class
modules for its elements, interactions, and views. A census on 2026-11-07 found 390 modules that register systems and
970 component modules: 784 serve one system, 39 serve several systems of one subfield, and 147 serve other components
or nothing. Six fields have a `common` package with 59 modules.

## Goals / Non-Goals

**Goals:**
- One module per system that holds everything only that system uses.
- Shared components of a field in its `common` package, and nowhere else.
- The same public names exported by every subfield package.
- Tests that mirror the new modules.

**Non-Goals:**
- Changing the behaviour, parameters, or registration of any system.
- Moving code into the core without the user's approval of the list of candidates.

## Decisions

- **Mechanical moves by a script, review by agents.** An AST-based script outside the repository builds the import
  graph of each subfield, folds every component module with one owning system into that system's module, rewrites the
  imports, and deletes the folded modules. Agents then review each touched module so it reads as one example, with
  classes ordered from elements to interactions to the factory, and fix what the script cannot, such as name clashes.
- **Ownership through chains.** A component imported only by other components belongs to the system that reaches it.
  When two systems reach it, it is shared.
- **Shared components.** A component shared by systems of one subfield moves to the field's `common` package when
  sharing makes sense, for example a population element that many ecology systems use. When the systems use it
  differently, each keeps a copy under a name that fits it. The agent decides per component and records the choice.
- **Existing `common` packages.** Their modules stay when several systems use them. A module used by one system moves
  into that system's module.
- **Cross-field duplicates.** After the moves, a script lists classes whose source is identical across fields. The
  list goes to the user with a proposal for each: move to the core under a mathematical name, or keep the copies.
- **Tests.** The test modules of folded component modules merge into the test module of the owning system. The layout
  test gains a check that no subfield module registers nothing while serving one system.
- **Waves.** One agent per field group, at most four at a time, each in its own worktree cut from `development`, in
  the order biology, physics, economics and social, then mathematics, engineering, computer science, and chemistry.

## Risks / Trade-offs

- **Merge conflicts with other work on systems:** the change starts after the open systems changes merge, and no other
  systems change runs while it does.
- **Long modules:** a system with many classes gets a long module. That is the intended trade, one complete example
  per file.
- **Lost history of moved code:** `git log --follow` does not follow a class into another file. The commit messages
  name the folded modules.
