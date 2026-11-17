# Design

## Context

`all_systems` lives in `skcomplex.systems._catalog`. Its private reader `_SystemClasses` scans the sources with `ast`
and imports only the modules of the classes it returns. Part classes carry no `field` or `topics`, but the class
rules give each kind a name suffix: `Element`, `Interaction`, `Law`, `Space`, `Neighborhood`, and `Time`. Every public
part is listed in the `__all__` of a package `__init__`, and many are exported again by a subfield package, such as
`PendulumElement`, which `systems.physics.mechanics` exports again from `systems.physics.common.mechanics`. Public class
names repeat across fields, such as `GrowthInteraction` and `CarElement`.

## Goals / Non-Goals

**Goals:**

- One function that answers "which part do I import" for every kind, from the core and from the systems packages.
- Selecting by kind and by package without importing other fields.

**Non-Goals:**

- No topics or tags for parts. Parts declare neither, and inventing them is out of scope.
- No change to `all_systems`, and no systems in the result.
- No split of `Element` into several classes. Its ports and agent hooks stay as they are.

## Decisions

- **Home: `skcomplex.systems`, next to `all_systems`.** The finder must see both the core and the systems packages,
  and the core never imports the systems packages, so it cannot live in the core. Keeping it in `_catalog.py` keeps
  the two finders side by side. A root module of `skcomplex` was the alternative, but it would add a second home for
  finders.
- **Name and arguments: `all_components(type_filter=None, package=None)`.** The code calls parts components, as in
  `BaseComponent` and `skcomplex.core.components`, and `type_filter` is the scikit-learn name. `package` takes the
  slash path below `skcomplex`, the same form `field` takes in `all_systems`.
- **Static reading by `__all__` and suffix.** A private reader, a sibling of `_SystemClasses`, walks the package
  `__init__` files under `skcomplex/core` and `skcomplex/systems`, reads `__all__` statically, and keeps the names whose
  suffix matches a selected kind. It then imports only the selected packages and checks that each class is a subclass
  of the base of its kind, `BaseElement`, `BaseInteraction`, `BaseLaw`, `BaseSpace`, `BaseNeighborhood`, or `BaseTime`.
  A name with the suffix that is not such a subclass raises `RegistryError`. Detecting subclasses with `ast` alone was
  rejected, since bases are often imported under other names.
- **Import paths, not bare names.** The result is `(import path, class)`. Bare names would collide, and the import
  path is the line the user writes. A class exported by several packages keeps the package whose path is the longest
  prefix of the class's `__module__`, which is the package that defines it.
- **Docstring of `Element`.** One paragraph on what an element is and how a subclass declares states and parameters
  with `Annotated`, `_get_laws`, and `law_classes`. One paragraph that sends the user to `all_components` first, with
  an example that finds `MassPointElement`. The required `Unlike`, `Choose it when`, and `It appears in` paragraphs
  stay. The ports paragraph is already covered by `ports_and_connections.md`. The rules on agent members, that a
  member is removed when its states tagged with a quantity reach zero and that states cannot take the names `location`,
  `alive`, `position`, or `system`, move to `agent_based_systems.md`, which does not state them yet.

## Risks / Trade-offs

- [Scanning every `__init__` is slow on first call] → Cache the static scan, as `_SystemClasses` does, and import only
  what is returned.
- [A public part without the class suffix is missed] → The class rules already require the suffix, and the convention
  check enforces it. A test checks that every exported subclass of a part base is found.
