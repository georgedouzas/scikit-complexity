# Design

## Context

The package `skcomplex.core.components` holds what a system is made of, and its root module `base.py` defines
`BasePart`, the base of `BaseElement`, `BaseInteraction`, `BaseLaw`, `BaseNeighborhood`, `BaseSpace`, `BaseTime`, and
`BaseSystem`. The docstring of `BasePart` says that "the parts of a system are its elements, interactions, laws,
spaces, times, and the system itself", and that sentence is repeated in many forms across the code, the guide, the
placement rule, and the specs. The guide section on the same classes is called "Core components", and the compile
passes already call the node of a flattened system its `component`. The maintainer approved one word, component.

`BaseComponent` is not a new name in this project. It was the name of the element base until the change
`2026-10-28-simplify-base` renamed it `BaseElement`. The archive keeps that history unchanged, and the main specs that
still use component in that older sense are reworded below, so the name now means the root base of every building
block.

## Goals / Non-Goals

**Goals:**

- One word, component, for a building block of a system, in public names, private names, messages, docstrings,
  documents, specs, plans, and the placement rule.
- Every other meaning of part and of component left as it is, and a written rule that tells them apart.
- A convention check that keeps part from coming back in the old sense.
- No published link that breaks silently.

**Non-Goals:**

- No alias for `BasePart` or `get_declaring_part`, and no test that pins their absence.
- No rename of the mathematical components, such as `components_of`, `component_names`, `n_components`, or the
  `components` reduction of a network.
- No change to the archive of finished changes.
- No change of behaviour. Every test that passes today passes after the rename with only names changed.

## Decisions

### The rule that tells the meanings apart

A word is renamed when it names a building block of a system: an element, an interaction, a law, a space, a
neighbourhood, a time, or a system, as a class, an instance, or a family. The test is substitution. If "an element, an
interaction, a law, a space, a time, or a system" can stand in its place and the sentence keeps its meaning, it is a
component. Otherwise it keeps its word.

Part keeps its other meanings:

- The pieces of a number or a tensor: `real_part`, `imag_part`, the real and imaginary parts of a complex state,
  `symmetric_part`, `antisymmetric_part`, `normal_part`, the linear and nonlinear parts of the rates in
  `core/engine/solvers/_split.py` with `_LinearPartsSpec` and `_get_linear_parts`, the divergence-free part of a
  Leray projection, the curl and harmonic parts of a Hodge decomposition, and the unique, redundant, and synergistic
  parts of a partial information decomposition.
- The places of a space: `PARTS = ('sites', 'edges', 'hyperedges')` in `core/components/elements/_group.py`,
  `TOPOLOGY_PARTS` and `SECOND_PARTS` in the compile passes, `_count_parts` of spaces, the `on=` argument of a group,
  and the requirement "Groups have a fixed or variable size and live on parts of the space".
- The pieces of a structure or a value: `delay_parts` and `_get_delay_parts`, the parts of a word or a patch of a
  rewriting system, the drive and relax items that `LatticeProblemPass._compile_part` compiles, `_split_part` and the
  stand in `_split_part_` of the complex pass, `PART_NAMES = ('real', 'imag')` of an entry spec, the `parts` of a
  linear program row, and the pieces of a string or a symbol name joined in printers and writers.
- The domain words of two models: the parts and subassemblies of the watchmakers of Simon in
  `systems/social/organizations/_watchmakers.py`, and `PartElement`, `PartView`, and `n_parts` of the technology
  designs in `systems/economics/innovation/_technology.py`.
- Libraries and docs tools: `shapely.get_parts`, and the `part` of a Rerun blueprint in `docs/gallery_tools`.
- Ordinary English: "in part", "takes part in", "is part of the trajectory", "the part `M` of the proliferating cells".

Component keeps its mathematical meanings, and in prose a mathematical component is always qualified by what it is a
component of or by its adjective, so that a bare "component" means a building block:

- Vector and tensor components: `components_of`, `get_components_of`, `component_names`, `vector(components)`, the
  components of a velocity, of a metric `(mu, nu)`, or of an alignment tensor.
- Connected components: the `components` reduction and `COMPONENT_REDUCTIONS` in
  `core/components/laws/utils/_reductions.py`, the Numba kernel `_numba_components.py`, and the giant component of a
  network.
- The `n_components` of decompositions and clusterings in `core/analysis`.
- The electrical components with ports of `systems/engineering/common/electrical/_components.py`. A resistor there is
  an element, so it is a component in both senses and needs no change.

### Inventory

The counts come from `rg` over the tree at the commit this change was drafted on, refined by a reader of the Python
syntax tree for names. A candidate is an occurrence of part or parts outside the words particle, partner, partition,
partial, participant, compartment, apart, and party. A candidate is kept when it matches one of the kept meanings
above, and the rest are renamed, each confirmed by hand per file.

`BasePart`:

| Where | Lines | Files |
| --- | --- | --- |
| `src` | 117 | 39 |
| `tests` | 25 | 4 |
| `docs` | 12 | 3 |
| main specs | 7 | 3 |
| constitution | 1 | 1 |
| drafted changes | 0 | 0 |
| archive, left as is | 44 | 24 |

Names in the code, counted as syntax tree names, grouped by package. Renamed:

| Package | Name | Uses | Files | New name |
| --- | --- | --- | --- | --- |
| `core/components` | `BasePart` | 43 | 21 | `BaseComponent` |
| `core/components` | `_get_part_specs` | 3 | 3 | `_get_component_specs` |
| `core/components` | `CompositeSystem._get_parts` | 2 | 1 | `_get_components` |
| `core/components` | `get_declaring_part` | 1 | 1 | `get_declaring_element` |
| `core/components` | `DoublingPart` of the doctest of the base | 5 | 1 | `DoublingComponent` |
| `core/components` | `part` and `parts` for a building block in `systems/_composite.py` | 21 | 1 | `component` and `components` |
| `core/engine` | `BasePart` | 10 | 2 | `BaseComponent` |
| `core/engine` | `_get_part_specs` | 1 | 1 | `_get_component_specs` |
| `core/engine` | `get_declaring_part` | 2 | 2 | `get_declaring_element` |
| `core/engine` | `FlattenPass._prepare_part` | 2 | 1 | `_prepare_component` |
| `core/engine` | `_SystemPartsSpec` | 27 | 4 | `_SystemVariablesSpec` |
| `core/engine` | `_system_parts_spec` module | 3 | 3 | `_system_variables_spec` |
| `core/engine` | `SystemDynamicsTranslator._read_parts` | 2 | 1 | `_read_variables` |
| `core/engine` | `parts` holding a `_SystemPartsSpec` | 101 | 3 | `variables` |
| `core/engine` | `part` for a building block in `_utils.py`, `_flatten.py`, `_flows.py`, `_census.py` | 30 | 4 | `component` |
| `core/analysis` | `get_declaring_part` | 1 | 1 | `get_declaring_element` |
| `core/analysis` | `part` for a building block in `_stability.py` | 6 | 1 | `component` |
| `systems/*` | `BasePart` | 22 | 16 | `BaseComponent` |
| `systems/biology`, `systems/chemistry` | `get_declaring_part` in doctests | 3 | 2 | `get_declaring_element` |
| `tests/core/components` | `BasePart`, `DoublingPart`, `ExtendedPart`, `InvalidSpecPart`, `SpecifiedPart` | 19 | 2 | the same names with `Component` |
| `tests/core/engine` | `BasePart` in `solvers/test_root.py` | 12 | 1 | `BaseComponent` |
| `tests/core/engine` | `_parts` of `translators/test_system_dynamics.py`, which returns the built system | 6 | 1 | `_composite_of` |
| `tests/core` | `BasePart` in `test_base.py` | 3 | 1 | `BaseComponent` |
| `tests` | test function names that say part for a building block | 19 | 10 | the same names with component |

Kept:

| Package | Names | Uses |
| --- | --- | --- |
| `core/components` | `PARTS`, `_count_parts`, `symmetric_part`, `antisymmetric_part`, `delay_parts`, the type `Part` and the `part` of a weighted term in `interactions/_coupling.py`, and the string pieces `parts` in `laws/utils/_reductions.py` | 61 |
| `core/engine` | `PARTS`, `Parts`, `SECOND_PARTS`, `TOPOLOGY_PARTS`, `_LinearPartsSpec`, `_get_linear_parts`, `_compile_part`, `_get_delay_parts`, `_split_part`, `delay_parts`, `real_part`, `imag_part`, `get_parts`, `_parts` of a linear program | 52 |
| `core/engine` | `part` and `parts` for complex parts, linear parts, and pieces of strings | about 145 |
| `core/specs` | `PARTS`, `PART_NAMES`, `EntrySpec._get_parts`, `parts` of patch structures and operators | 34 |
| `core/analysis` | `delay_parts`, the parts of a partial information decomposition | 18 |
| `systems/economics` | `PartElement`, `PartView`, `n_parts`, `part` | 13 |
| `systems/social` | `normal_part`, the `parts` of the watchmakers | 8 |
| `systems/physics` | `symmetric_part`, `antisymmetric_part`, `_get_dumbbell_parts` | 6 |
| `tests` | `PartsElement` of the complex pass, `_parts` of the information decomposition, the kept names above | 42 |

Prose candidates, as occurrences in docstrings, comments, messages, and pages, split into those the rule renames and
those it keeps. The files column counts the files with at least one rename.

| Where | Renamed | Kept | Files |
| --- | --- | --- | --- |
| `src/skcomplex/core/engine` | 411 | 149 | 56 |
| `src/skcomplex/core/components` | 302 | 53 | 55 |
| `src/skcomplex/core/specs` | 49 | 6 | 13 |
| `src/skcomplex/core/analysis` | 49 | 10 | 12 |
| `src/skcomplex/core`, other modules | 18 | 3 | 9 |
| `src/skcomplex/systems/biology` | 211 | 6 | 109 |
| `src/skcomplex/systems/physics` | 148 | 48 | 78 |
| `src/skcomplex/systems/economics` | 106 | 4 | 73 |
| `src/skcomplex/systems/social` | 99 | 3 | 50 |
| `src/skcomplex/systems/engineering` | 60 | 0 | 26 |
| `src/skcomplex/systems/mathematics` | 45 | 14 | 20 |
| `src/skcomplex/systems`, other fields and `general` | 42 | 3 | 24 |
| `tests` | 175 | 57 | 39 |
| `docs/overview/user_guide/core_components` | 171 | 16 | 8 |
| `docs/overview/user_guide`, other pages | 173 | 35 | 31 |
| `docs`, generators, hooks, and gallery | 79 | 31 | 27 |
| README, CONTRIBUTING, constitution | 27 | 2 | 3 |
| main specs | 89 | 33 | 22 |
| drafted changes | 100 | 8 | 47 |

These are the counts of the search before the review by hand, which moves some of them from one column to the other.
Two sentences make up most of the systems rows: `Return the parts of the system`, with 486 occurrences in 319
files, and `unique among the parts of its system`, with 140 occurrences in 109 files. Error messages that change
include `is not a part, so it declares no variables` in `core/engine/_utils.py`, `The parts ... disagree on the tag`
and `bind parts of one ...` in `core/components/systems/_composite.py`, `builds its parts as a CompositeSystem` in
`core/components/systems/_base.py`, `binds a part other than the field` in `core/analysis/_stability.py`, `the states
and parameters of its parts` in `core/experiments/_utils.py`, `values of parts that are affine` in
`core/experiments/_calibration.py`, and `a spec that the part handles` in `core/components/interactions/_base.py`.
No issue code holds the word, and the capability tags change only in the wording of `Tags` and its fields.

### Collisions and the wording chosen

- `_SystemPartsSpec` holds the stocks, flows, auxiliaries, constants, and lookups of a stock and flow file. In the same
  modules `AbstractComponent` of PySD and the local `component` mean one definition of a variable for a range of
  subscripts. Calling the spec components would give the word two meanings in four files. Its own docstring says
  "Variables of a system file", so it becomes `_SystemVariablesSpec`, its module `_system_variables_spec.py`, and the
  argument and attribute `parts` become `variables`.
- `core/engine/_utils.py` holds `declared_variables`, `declarations_of`, and `default_of`, whose argument `part` is a
  component class or instance, beside `components_of` and `vector(components)`, which read vector components. The
  argument becomes `component`, and the docstrings of the vector helpers say "the components of a vector" in full.
- `core/components/laws/utils/_reductions.py` defines the `components` reduction of connected components and builds
  call names from local `parts`, which are pieces of a string. The locals stay, and every docstring of the reduction
  says connected components in full.
- `FieldLowering._get_components` reads the vector components of a flux, while the compile passes call a node of the
  system its `component`. The method stays, and its docstring says vector components.
- `core/analysis/_stability.py` reads `components_of` a vector and reports an interaction that binds a part other than
  the field. The message becomes "binds a component other than the field", and the vector reads stay qualified.
- `systems/economics/innovation/_technology.py` uses parts and components for the pieces of a technology design, as
  the model of Arthur and Polak does. Its own names stay, and its system docstring says "Return the elements and
  interactions of the system" instead of the shared sentence.
- `systems/social/organizations/_watchmakers.py` counts the parts of a watch. Its own names stay, and its system
  docstring says "Return the elements and interactions of the system".
- Law utilities and interactions that apply an operation "to every component of a vector" keep that wording, which is
  already qualified.
- The guide page `build_a_custom_system/connected_components.md` assembles components joined through ports, while
  `analyse.md` and the `components` reduction use connected components in the sense of graph theory. The page becomes
  "Ports and connections" at `ports_and_connections.md`.
- The main spec `system-modeling` still says, from before `2026-10-28-simplify-base`, that "a system is components,
  interactions, space, and time", builds `CompositeSystem(components=[...])`, and that a system "is not a component"
  nor an instance of `BaseComponent`. With `BaseSystem` deriving from `BaseComponent` these become false. The first
  says elements and builds `CompositeSystem(elements=[...])`, and the second says that a nested system is a component.
  Its other uses, such as "laws of its own component", already fit the new sense and stay.

### Pages, generators, and published links

The generated pages under `parts/` describe nine families: elements, interactions, laws, spaces, systems, solvers,
backends, analyses, and experiments. The maintainer chose on 2026-11-11 to match the new word in the URL, so they move
to `components/<family>.md`, the nav section becomes "Components", the generator becomes `docs/generate_components.py`,
and the constants `PARTS` of the generator and of `docs/gallery_hooks.py` become `FAMILIES`, since a constant named
after components would read as the list of components. The pages of solvers, backends, analyses, and experiments sit
under the same path, and the index page of the section says that it covers the components and the tools that run and
study them. The link builder of `docs/generate_systems.py` points at `components/elements.md`. The alternative
`components/` was rejected by the maintainer.

The guide pages `core_components/choosing_parts.md` and `build_a_custom_system/connected_components.md` become
`choosing_components.md` and `ports_and_connections.md`. Their headings and the headings that say part, such as
"How the parts fit", "Where a part goes", "Where a new part goes", and "Two parts with one label", say component, and
every link to their anchors inside the documentation is updated.

The site is published at `https://georgedouzas.github.io/scikit-complexity`, and the README links the family pages by
their full URL. The eleven moved pages get redirects from `mkdocs-redirects`, a plugin of the MkDocs project whose
latest release, 1.2.3, is of March 2026 and which `properdocs` loads as an MkDocs plugin. Its pages keep the anchor of
the old URL. It enters the `docs` group of development dependencies, after `gen-files` in the plugin list so that the
generated targets exist. Writing redirect pages by hand in the generator was rejected, since a maintained plugin does
it. Links to the old anchors of renamed headings land at the top of the right page, which the impact section of the
proposal states. The strict build reports a broken link to a page today but only notes a broken anchor, so
`properdocs.yml` sets `validation.links.anchors` to `warn`, and the strict build then fails on any anchor the rename
misses.

### The convention check

`openspec/scripts/check_conventions.py` gains `check_components_are_not_called_parts`, listed in `CHECKS`, and the
self-check `check_part_names_are_named`, listed in `SELF_CHECKS`. The check reads:

- the names of the syntax trees of `src` and `tests`, which are classes, functions, arguments, assignments, and
  attributes, split into words at underscores and capitals, and flags a name with the word part or parts unless the
  whole name is in `KEPT_PART_NAMES`, a frozen set of the kept names of the inventory,
- the docstrings and comments of `src`, and the prose of the pages under `docs/overview`, the README, and CONTRIBUTING
  outside code blocks and inline code, and flags a line with the word part or parts unless the line matches one of
  `KEPT_PART_PHRASES`, compiled patterns of the kept meanings, such as a qualifier before part, "parts of" a space, a
  lattice, a network, a word, a patch, a value, or a matrix, "in part", "takes part", "is part of", and a fraction in
  backticks after "the part".

Each finding reads `file:line: part names a component: <text>`. The self-check writes a planted module with a class
`DoublingPart(BaseComponent)`, a docstring `Return the parts of the system.`, a call of `real_part`, and the phrase
"its real part", and a planted page with "every part of a system" and "takes part in", and expects exactly the class,
the docstring, and the page line, as the existing self-checks do. The check `check_library_generates_no_classes_of_parts`
becomes `check_library_generates_no_classes_of_components`. The README of the scripts lists the new check in its
sentence of examples.

### Tests

Tests mirror the sources. Renamed test classes and test functions stay in their files, and no test file is added. The
module `_system_parts_spec.py` has no test module of its own today, so its rename adds none. No test checks that
`BasePart` or `get_declaring_part` is gone, since the maintainer forbids tests that pin history. Pinned messages that
say part, such as the tag disagreement in `tests/core/components/systems`, are updated with the message.

### Memory and process

After this change the agent briefs, the memory notes of the maintainer, and the placement rule text say component for
a building block of a system. The notes on the placement rule, on the API goal, and on scikit-learn alignment say
"estimators for components" and "functions for what is done to components". A brief that says part in the old sense
is corrected before an agent starts.

## Order of work

1. Code: the public names, then the private names and the messages by package, `core/components` first, then
   `core/engine`, `core/analysis`, `core/experiments`, `core/specs`, and the fields of `skcomplex.systems`.
2. Tests: the names and the pinned messages, in the order of the code.
3. Docs: the generators and the hooks, `properdocs.yml`, the guide pages and their links, the glossary, the
   troubleshooting page, the README, and CONTRIBUTING.
4. Specs: the constitution, the main specs through the delta specs of this change at archive time, and the purpose
   of `general-systems`, which a delta cannot change.
5. Drafted changes: the prose of each draft under `openspec/changes`, archive excluded.
6. Check: the new convention check and its self-check last, so that it runs on the finished tree and its allow list
   holds only what the rule keeps.

## Risks / Trade-offs

- A missed rename leaves part in the old sense. The review is by hand per file, with the inventory as the list of
  files, and the new check fails on any name or line the review missed.
- A wrong rename turns a real part or a vector component into a component, which changes a meaning or, for a name,
  breaks a kept name such as `real_part` in a law. Sed is not used across files. Kept names are listed in the allow
  list before the check runs, the doctests and the markdown blocks run every example, and the tests run in full.
- Two words for vector components and building blocks remain in the same files. The rule that a mathematical
  component is always qualified keeps the bare word for building blocks, and the collisions above name each file.
- The allow list of phrases can grow until it hides an old use. Each phrase names a meaning of the rule above, and a
  phrase that only fits one sentence is rejected in review.
- Old anchors on the published site land at the top of their page. The pages themselves redirect, and anchors inside
  the site are checked by the strict build.
- Code outside the repository that imports `BasePart` breaks. The breaking commit and the release notes say what to
  import instead.
