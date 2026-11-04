# Design

## Context

Observed in the code on 2026-11-04:

- `RoleSpec` holds `variadic`, `noise`, and `optional`, and nothing about what fills the role.
- Interactions return their laws from `_get_laws(s)`, so what a role needs is only found by running it. With a missing state
  the compile raises `AttributeError` from the namespace.
- `system-validation` already has one pass that reports every problem, with a code, a path, and a fix, for unbound roles and
  for laws in the wrong time. `docs/generate_compatibility.py` builds the page from the declarations of laws, solvers, and
  backends.
- `catalog`, `describe`, `create`, and `run` exist, the Models tab has a page per model, and the page already prints the source
  of a model's components, though not as a list of what is compatible.
- Of the 59 field-module imports in the guide, 21 are `biology.epidemiology` and 11 `mathematics.fractals`, used as convenient
  parts for examples of other features.

## Goals / Non-Goals

**Goals:**

- A reader finds the right path in the first screen of the README and of the guide.
- A user can answer "does this interaction fit this element?" from a page, from code, and from the error when they get it wrong.
- The answers are generated from declarations, so they cannot drift.

**Non-Goals:**

- Moving code between packages. Realizations stay in `models`.
- Rewriting gallery examples whose subject is a model's own analysis or parts.
- A graphical model builder.

## Decisions

### Which imports count as custom

A custom model imports from `skcomplex.core`, `skcomplex.specs`, `skcomplex.engine`, and `skcomplex.analysis`. Everywhere
in the README and the guide, `skcomplex.models` appears only as `from skcomplex.models import catalog, describe, create, run`.
A guide section that documents a model family, such as Metabolism, selects its models by name. A test reads the Python
blocks of both files and fails on any import from a field module.

Examples that used `PopulationElement` define a small `Element` in a few lines, which also teaches the concept. To keep the
guide readable, one such element is defined early, and later blocks continue from it.

### Declared requirements, with a fallback

`RoleSpec` gains `elements`, the element kinds a role accepts, and `states`, the state names it reads. A name that is an
argument of the constructor means the value of that argument, because `RateInteraction(state='x')` reads what it is told.

Inferring the reads by running `_get_laws` with a recording namespace was rejected as the primary source: it is implicit,
needs a built system, and cannot state element kinds, so it answers nothing before the user has chosen. It remains useful in
one place. Interactions that declare nothing, such as a user's own, are still covered by catching the missing attribute while
building their laws and reporting it as the same issue, naming the role and the state. So the bare `AttributeError` goes away
for every interaction, and the declarations add the answer in advance.

### Public class methods

The project keeps public functions out of the packages, so this is on the classes, in the style of `get_params`:

- `BaseInteraction.get_requirements()` returns the requirements by role, and the laws' needs of time and space where they
  are declared.
- `BaseElement.get_states()` returns the states an element declares with their specs.

`add-declared-state-elements` makes the states of an element possibly a fact of an instance, so `get_states` is an instance
or class method to match how that change settles it. This change applies after it.

### One page for compatibility

The generated page keeps its sections and gains two, elements and interactions, listing for each interaction its roles, the
element kinds and states each needs, and the laws it returns. Where an interaction's laws can only be known by running it,
the page says so rather than guessing. The guide's new section links to it, and walks the order of choices: time and space,
then elements, then interactions, then the solver.

### Catalog in the documentation

The Models tab gains an index page with every model by field and tag, with the description, generated from `catalog()`, and
static client-side filtering only if the site's search does not already serve it. Each model page gains a short block on
selecting it with `create` and `describe`, and the list of element and interaction classes its system uses, read from the
system the page already builds.

## Risks / Trade-offs

- [The guide is 7422 lines, and other changes are editing it] → This change applies after the open changes, and rewrites the
  guide section by section, with the guide's blocks tested throughout.
- [42 interactions need declarations] → Each is one annotation, and a test checks that every core interaction declares its
  roles. A wrong declaration is caught by running the catalog through validation: no prebuilt model may report the new issue.
- [`AttributeError` from a genuine bug in a law could read as a missing state] → The fallback reports only a missing attribute
  on a role's namespace, and leaves every other error as it is.
- [Requirements could reject a valid use] → The kinds default to any element, so a declaration only narrows what is true.

## Migration Plan

Declarations and validation land first, since they change no valid system. Then the generated pages, then the rewrite of the
README and the guide. Each step keeps the documentation build and the guide's tests passing.

## Open Questions

- Whether the README's first custom example stays predator and prey, written from the core, or becomes smaller. Predator and
  prey is proposed, since a reader knows it and it shows two elements and two kinds of interaction.
