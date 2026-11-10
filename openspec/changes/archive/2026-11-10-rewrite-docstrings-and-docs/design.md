## Context

See proposal.md for the motivation. The style rules already in force are these.

- The constitution asks for one style written for a reader who wants to understand, sentences in the normal order,
  no passive voice without reason, a one-line summary, an `Args` block for every parameter, and a runnable example
  for public API. A module docstring is one imperative line.
- `pyproject.toml` sets the Google convention for ruff's pydocstyle rules. pydoclint checks the Google style against
  the signatures, without types in the docstring, without class attributes, and with constructor docstrings allowed.
  interrogate checks docstring coverage. These stay as they are.
- The spec `user-documentation` bans semicolons and hard-coded counts of the catalog in the docstrings and the
  documents. The spec `package-layout` runs every docstring example through pytest's doctest collection, and the
  README and the user guide through `pytest-markdown-docs`.
- `tests/core/components/test_base.py` holds `FIELD_TERMS`, a list of field words that no public name of
  `skcomplex.core.components` may contain, and walks the modules of the core packages.
- The user's style edits of 2026-11-08 and 2026-11-09 add short summary lines, "main public methods" in place of
  "verbs", and plain docstrings with cross-domain examples.

Scope, measured with an AST walk of `src/skcomplex/core` on 2026-11-09, before the earlier changes land. The numbers
are a snapshot for planning and go in no document.

| Package | Modules | Public classes | Public methods | Public functions |
| --- | --- | --- | --- | --- |
| `components` root (`base`, `types`) | 3 | 3 | 43 | 0 |
| `components/elements` | 12 | 11 | 5 | 0 |
| `components/interactions` | 45 | 44 | 11 | 0 |
| `components/laws` and `laws/utils` | 49 | 30 | 3 | 66 |
| `components/spaces` | 28 | 29 | 0 | 0 |
| `components/specs` | 12 | 10 | 13 | 0 |
| `components/systems` | 4 | 4 | 1 | 0 |
| `specs` | 132 | 115 | 115 | 6 |
| root (`base`, `errors`, `types`) | 4 | 42 | 3 | 0 |
| `analysis` | 56 | 53 | 4 | 53 |
| `experiments` | 16 | 6 | 6 | 8 |
| `engine` root, `runtime`, `viz`, `backends`, `translators` | 47 | 35 | 23 | 10 |
| `engine/solvers` | 53 | 52 | 55 | 0 |
| `engine/compile` | 82 | 75 | 8 | 1 |
| `engine/problems` | 80 | 65 | 82 | 1 |

In total about 620 modules, 570 public classes, 370 public methods, and 145 public functions. Of the class
docstrings, about 50 run past 150 words before their first section, and the longest run to 700 or 800 words
(`TransferInteraction`, `CouplingInteraction`, `SamplingInteraction`). More than half of the module docstrings have a
body. Few summary lines break the mechanical rules today: a handful run past 80 characters, and four docstrings hold a
semicolon. The main fault is the content, not the length of the first line.

## Goals / Non-Goals

**Goals:**

- One set of testable rules for docstrings and documents, written down here and in the spec.
- Prose that reads as written by a careful person.
- A full reference of the core components for builders of systems.
- Tutorials for every workflow of a system, and case studies that run the whole workflow in several fields.
- Pilots the user approves before the bulk of the work.
- A light guard for the rules that a machine can check.

**Non-Goals:**

- No change of a name, a signature, or behaviour. A docstring that reveals a confusing name is noted for a later
  change, not renamed here.
- No style linter, no new dependency, and no change to the pydoclint, interrogate, or ruff settings.
- No new public API. A tutorial that finds a missing function notes it for a later change.
- Private functions and methods keep their one-line summaries and are only checked for length and semicolons.
- The docstrings of `skcomplex.systems` are out of scope, except `skcomplex.systems.general`. They follow the same rules
  when a later change touches them. `skcomplex.systems.general` holds the cross-domain recipes that
  `move-recipes-to-systems` moves out of the core, whose docstrings this change was already to rewrite, so they stay in
  scope and follow the rules of the core components.

## Decisions

### The rules

1. **Summary.** One line, at most 80 characters, ending with a full stop. A class summary starts with "A" or "An" and
   names its kind: "An element that holds one nonnegative number." A function or method summary is imperative:
   "Return which pairs of states of a run lie close to each other." No summary holds a field term of `FIELD_TERMS`.
2. **Body.** One to three short paragraphs in plain words. The first says what the thing does. A list may name the
   modes of an argument such as `mode` or `kind`, one line each. The details of a mode go to its `Args` entry, not to
   the body.
3. **Uses.** For a cross-domain part, which is every public class of `components`, every analysis function, and every
   experiment function, a paragraph names concrete uses from at least two fields, two to four of them, in one or two
   sentences. A solver, a backend, and a spec name uses only where the user picks them by hand. The other engine
   classes skip this paragraph. The paragraph starts with "It appears in", so the generator of the parts pages can
   read it.
3a. **Comparison.** The docstring of every public class of the core elements, interactions, laws, spaces,
   neighbourhoods, times, and systems holds a paragraph of two sentences. The first starts with "Unlike" and names
   what the class adds over its nearest part. The second starts with "Choose it when". For example: "Unlike
   `Element`, it needs no subclass and keeps its one state at zero or above. Choose it when a system tracks an amount
   that cannot be negative, such as a count, a mass, or a sum of money."
4. **Args.** Each entry says what the value means, its units, its range or its allowed values, and an example value
   where it helps. It does not say how the code stores, converts, or passes the value on.
5. **Jargon.** A field term or an internal term, such as a class axis, an attractor, a Theiler window, or a law
   lowering, gets a plain definition in the same docstring. The text says "main public methods", never "verbs".
6. **Examples.** Every public class and function keeps a runnable `Examples` section. Where a cross-domain use fits
   in a few lines, the example builds it, such as an epidemic compartment beside a bank account. Otherwise the uses
   stay in the prose. A short sentence may open a group of examples to say what it shows.
7. **Plain text.** No semicolons, no passive voice without reason, no rhetoric, and no hard-coded counts of the
   catalog. Sentences are short and in the normal order.
8. **Module docstrings.** A private module has one imperative line that names its concept: "Define the element that
   holds one nonnegative number." A package `__init__` has one imperative line and at most one short paragraph that
   names what it holds.
9. **Documents.** The rules 5 and 7 apply to the README, the user guide, the generated pages, `CONTRIBUTING.md`, and
   the text of the gallery examples. Each section opens with what the reader achieves, in one or two sentences. Every
   concept is shown in a code block that runs, placed right after the paragraph that first names it. The tone is
   academic but simple: precise words, full sentences, no slang, no exclamation marks, and no words such as "just",
   "simply", or "magic".
10. **Natural prose.** The text reads as written by a careful person, not generated.
   - No semicolons and no dashes as punctuation: no em dash, no en dash, and no hyphen between spaces. A clause that
     wants a dash becomes its own sentence, or joins with a comma, "and", "so", or "because". Hyphenated words such as
     "cross-domain", Markdown list markers, table rules, and code stay allowed.
   - No stock openers, closers, or filler. The banned list holds "in summary", "in conclusion", "to summarize", "it is
     worth noting", "it's worth noting", "note that", "let's dive", "dive into", "delve", "leverage", "seamless",
     "seamlessly", "robust", "powerful", "cutting-edge", "unlock", "harness", "empower", "plethora", "myriad",
     "a wide range of", "in today's", "simply", "just", "easily", and "magic". Code, inline code, and the titles of
     cited references are exempt.
   - No rule-of-three padding. A list of three appears only when there are three things.
   - A paragraph, not a list, where the items need connecting words. A list holds parallel items that a reader scans.
   - Sentence length varies. A short sentence follows a long one where the point is simple.
   - Precise verbs: "integrates", "draws", "sums", "raises", not "handles", "deals with", "manages", or "processes".
   - Every claim is backed by the code on the page or by a cited reference. A page does not praise the library.

Alternative considered: a third-party prose linter such as Vale. It needs its own rules and configuration, and the
user asked for a light guard, not a style linter.

### The approved style

The user approved the pilot of groups 1 and 2 on 2026-11-10 without corrections. The rewritten package
`skcomplex.core.components.elements` and `CouplingInteraction` are the reference for every later group. A class
docstring holds these parts, in this order.

1. The summary: "An element whose states and parameters are given as data." It names the kind after an article.
2. The body, one or two paragraphs that say what the part does in plain words, defining every term it uses. A mode
   list, one line per mode, follows the paragraph that names the argument. Details of a mode go to its `Args` entry.
3. The comparison: "Unlike `Element`, it needs no subclass, since it takes its states and parameters as data. Choose it
   when ..." It names the nearest part and what the class adds over it.
4. The uses: "It appears in many fields." followed by concrete uses in one or two sentences, such as people infected in
   an epidemic or money in a bank account. A part with fewer honest uses names those it has, such as "It appears in
   lattice models of ...", and never invents one.
5. `Args`, each entry a meaning with an example value, such as "The name of the element, unique among the parts of its
   system, such as `'infected'`", and a list of allowed values where the argument takes a fixed set.
6. `Attributes` and `Raises` where they apply, with `Raises` saying which main public method raises the error.
7. `Examples` that open with one sentence on what they show, then build a use from one field and, where it fits in a
   few lines, a second field with the same class.

A method keeps an imperative summary, `Args`, `Returns`, and a short example. A private module has one imperative line
that names its concept, such as "Define the interaction that couples each member to its neighbours.", and a package
`__init__` adds one short paragraph that names what it holds.

The user approved the documents pilot on 2026-11-10: `core_components/index.md`, `core_components/elements.md`, and
the build tutorial. They are the reference for every later page. A family page opens with what the family is for and
links the page on how the parts fit and the parts page. Each class has a `## ClassName` section in this order: a
paragraph on what it models with the first runnable example, the same class in another field with a second example,
then the paragraphs **How the engine runs it.**, **How it combines.**, **Limits and errors.** with one example that
catches the error into `message`, and **A field realization.** with a short subclass, then one line that links its fact
sheet and its API entry. Code blocks end on an expression and print nothing by hand. The prose after a block states
the values the block returns, taken from a real run. The pelt records of the build tutorial are those distributed
with Carpenter (2018) in the Stan example models, and the runs start from the records of 1900.

### The guard

Four tests are added to `tests/core/components/test_base.py`, which already walks the core and holds `FIELD_TERMS`.
They walk every module of `skcomplex.core`, with `skcomplex.core.specs` and the root modules added to the walk, and
read the docstrings of modules, public classes, public functions, and public methods.

- `test_docstring_summaries_are_short`: the first line is at most 80 characters.
- `test_docstrings_read_naturally`: no prose line holds `;`, an em dash, an en dash, a hyphen between spaces, or a
  phrase of the banned list of rule 10, compared case-insensitively on whole words. Lines that start with `>>>` or
  `...`, and inline code between backticks, are exempt. The banned list is a module constant `BANNED_PHRASES` beside
  `FIELD_TERMS`.
- `test_class_summaries_name_their_kind`: the summary of a public class starts with "A " or "An ", and contains the
  kind word of its name suffix. The map of suffix to word is `Element: element`, `Interaction: interaction`,
  `Law: law`, `Space: space`, `Time: time`, `Neighborhood: neighbourhood`, `System: system`, `Solver: solver`,
  `Backend: backend`, `Observer: observer`, `Spec: spec`, `Result: result`, `Pass: pass`, `Error: error`. A class
  whose suffix is not in the map is checked for the article only. No summary of a public class or function holds a
  word of `FIELD_TERMS`, compared case-insensitively.
- `test_docstrings_say_main_public_methods`: no docstring holds the word "verb" or "verbs".
- `test_components_compare_and_name_uses`: the docstring of every public class of the core component families holds
  a paragraph that starts with "Unlike" and holds "Choose it when", and a paragraph that starts with "It appears in".

The first two hold for the whole core once the current findings are fixed in the first task group. The docs hold
dozens of en and em dashes and spaced hyphens today, so the documents test starts with the README and the pages
already rewritten, through a tuple `REWRITTEN_DOCUMENTS` in `tests/test_sources.py`, and the last group removes it. The other
three hold only after a package is rewritten. So they read a module constant `REWRITTEN`, the tuple of package prefixes
already rewritten, which each task group extends. The last group removes the constant and checks the whole core.

`tests/test_sources.py` gains `test_documents_read_naturally`, which reads the README, `CONTRIBUTING.md`, the pages of
the user guide, and the docstrings of the gallery examples. It skips fenced code blocks, inline code, Markdown list
markers, and table rules, and fails on a prose line with a semicolon, a dash used as punctuation, or a banned phrase.
It imports nothing from the core tests and holds its own copy of the banned list, since a test module must not import
another. The two lists are kept equal by a third test that reads both constants as text.

`tests/test_sources.py` also gains `test_core_components_pages_cover_every_class`. It reads the `__all__` of each family
package and fails when a public class, other than an abstract base, has no `## ClassName` heading on the page of its
family. It reads the `__all__` of every concept subpackage of `skcomplex.systems.general` against
`core_components/general.md` the same way, and it fails when a code block of a family page imports from
`skcomplex.systems`, since the family pages document the core alone. `test_troubleshooting_covers_every_error` does the
same for the public error classes of `skcomplex.core.errors` and the troubleshooting page.

No test can judge flow. Each page is also read through once, aloud in style, before it is shown to the user: the
reviewer reads every sentence in order, marks sentences that sound generated, stiff, or padded, and rewrites them. The
tasks hold one such review per page group.

Alternative considered: one guard test module under `tests/`. The constitution asks the test tree to mirror the
source tree, and `test_base.py` already holds the cross-cutting checks of the core, so the guard joins it.

### Only the public interface is documented

The user stated that a user is shown only public methods. Today about forty docstring example lines in the core and
the systems, and several guide pages, call private methods such as `_get_laws(...)`, `_get_components_of(...)`, or
`_reduce_pairs(...)` to show internals. Each example is rewritten to show the public result, for example the
trajectory or the `laws_` attribute after `run`, or the example is removed and its check kept in the tests.
The one allowed form is defining a documented extension hook in a subclass on the extension pages and in base class
docstrings, because that is how a builder writes a new part. Calling a hook in an example is never allowed. The
guard `test_documents_show_only_public_names` walks every docstring example (`>>>` and `...` lines) in `src` and every
code block of the README, the user guide, and the gallery, and fails on an attribute access or call whose name starts
with an underscore, except a `def _name` inside a class body and dunder names.

### The structure of the guide

The guide keeps its task-led order and grows into five parts. Files are named by what they hold.

```text
docs/overview/user_guide/
  index.md
  select_a_ready_made_system.md, change_it.md, system_classes.md      getting started, kept
  core_components/
    index.md            how the parts fit, diagrams, a worked example (absorbs concepts.md)
    elements.md, interactions.md, laws.md, spaces.md, systems.md       the reference, one page per family
    general.md          the recipes of skcomplex.systems.general and the placement rule
    choosing_parts.md   the decision guide
  build_a_custom_system/
    index.md            the build tutorial
    lattice_systems.md, networks.md, ...                               recipes by kind of system, kept
  check.md              new, absorbs choosing_compatible_components.md
  simulate.md, analyse.md                                              rewritten as tutorials
  evaluate.md, calibrate.md                                            new
  run_experiments.md    the experiment tutorial
  observe.md            the visualise tutorial
  share.md              new, beside extend_the_library.md
  case_studies/
    index.md, epidemic_calibration.md, opinion_sweep.md, ising_transition.md
  glossary.md, troubleshooting.md
```

The navigation in `properdocs.yml` follows this order, which the modified requirement "The user guide is task-led"
fixes. `concepts.md` merges into `core_components/index.md` and `choosing_compatible_components.md` into `check.md`,
so every earlier topic keeps a section and none is told twice. Links to the two merged pages are updated, and the
strict build catches any that remain.

Alternative considered: a separate "Tutorials" tab beside the guide. It splits one reading order in two, and the
spec asks for one task-led guide.

### The core components reference

The reference is for builders of systems, so it explains every core class in full. Each fact has one home.

The family pages document the core building blocks only, with the classes that stay in the core after
`move-recipes-to-systems`. Their examples import nothing from `skcomplex.systems`, as the core itself never does.
`NonnegativeElement`, `PointElement`, `BirthDeathInteraction`, `EncounterInteraction`, `ReactionNetworkSystem`, and
the other recipes leave the core, `PairCouplingInteraction` merges into `PairInteraction`, `ConeNeighborhood` merges
into `RadiusNeighborhood` with an angle, and `PseudoRiemannianSpace` leaves for physics as `SpacetimeSpace`. An
example that used one of them is rebuilt from parts that stay, or moves to the page of `skcomplex.systems.general`.

**The page of `skcomplex.systems.general`, in `core_components/general.md`.** It documents the cross-domain recipes
and the general systems that several fields share, one `## ClassName` section per public class of each concept
subpackage (`elements`, `agents`, `matching`, `kinetics`, `reactions`, and `dynamics` once `add-general-systems` lands),
with the same section template as the family pages. Each section names the core parts the recipe is written with, so a
reader sees, for example, that `ReactionInteraction` is a `JumpInteraction` with stoichiometric changes. The page
closes with the placement rule of `move-recipes-to-systems`, quoted word for word, and one worked example per step: a
structural base, a part the engine handles by type, a recipe of several fields, one field, and one system, and a
recipe that reveals a missing primitive. The page on shared realizations that `move-recipes-to-systems` writes becomes
this page.

**Hand-written, in `core_components/<family>.md`.** Each family page opens with what the family is for and how it
fits with the other families. Each public class then has a `## ClassName` section with these parts, as short as the
class allows:

- What it models, in one plain paragraph, with the first runnable example.
- The same class in another field, with a second runnable example. The examples use the core alone. Where a field
  realization exists, the prose names it after the example.
- How the engine runs it: which laws it emits, and what a solver does with them, such as integrating a rate or
  applying an update once per step.
- How it combines with other parts: the roles it fills or binds, and its usual partners.
- Limits and errors: what it cannot express, and one example that triggers its most common error and prints the
  message, followed by the fix.
- A field realization: a short subclass that renames the part for one field, as the catalog does.
- A link to its fact sheet and to its API entry.

**Generated, in `parts/<family>.md`.** `docs/generate_parts.py` writes, per family, the comparison table and one fact
sheet per class. A fact sheet lists the states with their domains, the roles with the element kinds and states they
accept, the parameters, the law kinds it emits, the times, solvers, and backends whose tags accept it, the errors of
its `Raises` section, and links to its reference section, its API entry, and the catalog systems that use it. The
meaning of each state, role, and parameter comes from the `Args` section of the docstring, parsed with griffe, the
docstring parser that mkdocstrings already uses. Its package `griffelib` is already installed, and the docs group of
`pyproject.toml` names it because the generator imports it. A new parts page covers spaces, neighbourhoods, and times. The tag
tables and the compatibility examples stay.

**Catalog links.** A system uses a class when its module, or a module of its field `common` package that it
imports, defines a subclass of the class or names it in its globals. The generator finds them by walking the
registered systems of `skcomplex.systems`. Subclasses are shown as field realizations, such as `CellAdhesionInteraction`
of biology for `SamplingInteraction`. `docs/generate_systems.py` writes the reverse links: each catalog system page
lists the core classes it builds from, each linked to its fact sheet. The links are built at build time, so no count
goes stale.

The docstring is the one source of what a class is, what it adds, when to choose it, its uses, the meaning of its
arguments, and its errors. The code is the source of its states, roles, laws, and tags. The reference page adds what
neither can hold: how the class behaves in a run, how it combines, and examples from several fields side by side.

Alternatives considered: hand-written pages alone, which copy tags and argument lists that then go stale, and
generated pages alone, which cannot explain behaviour and whose `markdown-exec` blocks do not run in the test
session.

**How the parts fit.** `core_components/index.md` opens the reference. It explains that an element holds states, an
interaction reads them through roles and emits laws, a law is one equation or rule for one state, a space places the
members of a group and says who neighbours whom, a time says whether the system steps or flows, and a system composes
them. Two diagrams in Mermaid show it. The first is a class diagram of the parts and their relations. The second is a
flow chart of a run: `prepare` checks the parts and collects their laws, the compiler lowers the laws to a problem,
a solver advances the problem on a backend, and observers record a trajectory. A worked example then builds one
system in steps, a predator and its prey on a lattice, running after each step: the elements, then one interaction,
then the space, then the time, then the system, then `simulate`. A table at the end maps each family to its import
path.

Mermaid needs one entry in `properdocs.yml`: a custom fence `mermaid` under `pymdownx.superfences` with
`pymdownx.superfences.fence_code_format`, which the Material theme renders. No new dependency is added.

The examples of the reference, among others:

| Class | Example in one field | Example in another field |
| --- | --- | --- |
| `DeclaredElement` with `RateInteraction` | an epidemic, infected people recover at a rate | a bank account, interest grows the balance |
| `PairInteraction` with a potential | atoms of a gas under the Lennard-Jones potential | pedestrians that keep their distance |
| `GroupElement` | a herd of grazing animals | a portfolio of assets |
| `CouplingInteraction` | fireflies that flash in step | consensus of opinions on a network |
| `JumpInteraction` | births and deaths of a population | customers who join and leave a queue |
| `ArrayUpdateLaw` | grains added to a sandpile | a random mutation of a genome |
| `BallNeighborhood` | toppling sites of a sandpile | the neighbours of a voter model |
| `NetworkSpace` | an epidemic on a contact network | a power grid of coupled generators |
| `CompositeSystem` | a predator and its prey on a lattice | a savings account with interest and withdrawals |

The examples that the first draft gave with `NonnegativeElement`, `PointElement`, `BirthDeathInteraction`, and
`MassActionSystem` move to `core_components/general.md`, where `NonnegativeElement`, `PointElement`,
`BirthDeathInteraction`, and `ReactionNetworkSystem` keep the same pairs of fields.

### The workflow tutorials

Each workflow page is a tutorial with one running example: the hare and lynx populations of the Hudson's Bay Company
pelt records, the classic data set of predator and prey cycles. The build tutorial builds it from core parts and then
as a subclass of a general system. Every later page starts from the catalog class `LotkaVolterraSystem` in one line,
so each page runs alone in the test session. The data sit inline in a short array with their source cited.

- **Build** (`build_a_custom_system/index.md`): compose elements, interactions, and a system, then write the same
  model as a subclass of `ReactionNetworkSystem` of `skcomplex.systems.general.reactions` and as a `BaseSystem`
  subclass with model parameters.
- **Check** (`check.md`): `prepare`, `get_tags`, the requirements of an interaction, and the validation errors a
  builder meets most, each triggered, printed, and fixed.
- **Simulate** (`simulate.md`): `simulate` with its run settings, the choice of solver and backend from the tags,
  `random_state` for reproducible stochastic runs, and the `Trajectory` it returns.
- **Analyse** (`analyse.md`): the analysis functions that apply to the example, found with `applicable_analyses`,
  and how to read each result.
- **Evaluate** (`evaluate.md`): comparison with the pelt records, the goodness of fit of a calibration, posterior
  predictive checks, `posterior_coverage`, and practical identifiability with `identifiability`.
- **Calibrate** (`calibrate.md`): every method of `calibrate`, which are `'local'`, `'global'`, `'abc_smc'`,
  `'mcmc'`, and `'variational'`, with the losses each takes, its diagnostics, and when to choose it.
- **Experiment** (`run_experiments.md`): `sweep`, `ensemble`, `sensitivity`, `compare_scenarios`,
  `basin_stability`, and `orbit_diagram`.
- **Visualise** (`observe.md`): the `RerunObserver` with a blueprint, the visuals of `skcomplex.core.engine.viz`, and
  plots of trajectories and results with matplotlib.
- **Share** (`share.md`): turn the example into a system class, register it, add its inventory entry and gallery
  example, and contribute it, with links to `CONTRIBUTING.md`.

Expensive methods run with small budgets and fixed seeds, so each page stays fast in the test session. The prose says
which budget a real study would use.

### The case studies

Three case studies from different fields take one system through the whole workflow, each against data or a known
result.

- **An epidemic calibrated to case data** (biology): `SIRSystem` calibrated to the daily counts of the 1978 influenza
  outbreak in an English boarding school, reported in the British Medical Journal. It checks the fit and the posterior
  coverage, studies the identifiability of the two rates, and compares scenarios of earlier isolation.
- **An opinion model swept and analysed** (social science): a bounded confidence model swept over the confidence
  bound, with an ensemble per value, the number of opinion clusters from `clusters`, and the sensitivity of the
  final consensus to the bound and the convergence rate.
- **A lattice phase transition** (physics): the two-dimensional Ising model swept over temperature on lattices of
  several sizes. It locates the peak of the susceptibility near the exact critical temperature of Onsager,
  `2 / ln(1 + sqrt(2))`, and estimates critical exponents with `scaling` and `data_collapse`.

Each case study cites its data or its exact result. Data enter as short inline arrays, so the pages need no download.

### Extras

- **Glossary** (`glossary.md`): one entry per mathematical or library term the guide uses, such as attractor, law,
  role, tag, mass action, phase space, and posterior. Each entry is two or three sentences with a link to where the
  term is used. A docstring still defines a term where it uses it, as rule 5 asks.
- **Decision guide** (`core_components/choosing_parts.md`): Mermaid flow charts that lead from the features of a
  problem to parts. Does the state flow or step? Is it one amount, a vector, or many members? Do members sit on a
  lattice, a network, or in continuous space? Do they act by rates, maps, or agent rules? Each leaf links to the
  reference sections of its parts.
- **Troubleshooting** (`troubleshooting.md`): one section per public error class of `skcomplex.core.errors`. Each
  shows the shortest code that raises it, the message, what it means, and the fix. The docstring of each error class
  says when it is raised and how to fix it, and the page shows it happen.
- **Cross-links**: the reference and the tutorials link each core class to its API entry with autorefs, fact sheets
  link the reference and the catalog, and catalog pages link the core classes they build from. The strict build
  fails on a broken link.

#### A sketch of the elements page

````text
# Elements

An element holds the states of a system, the numbers that change as the system runs. An interaction reads these
states and emits laws that change them. A space places the members of a group, and a system composes elements,
interactions, and a space into one model. The table on the [elements parts page](../../../parts/elements.md) compares
every element class.

## DeclaredElement

A `DeclaredElement` takes its states and parameters as data, so it needs no subclass. The same element class models
people in an epidemic and money in a bank account.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(100.0, StateSpec())})
recovery = RateInteraction('recovery', target='infected', state='x', rate=lambda s: -0.1 * s.target.x)
epidemic = CompositeSystem(elements=[infected], interactions=[recovery])
simulate(epidemic, until=10, dt=5.0).values('infected.x').round(1).tolist()
```

```python continuation
account = DeclaredElement('account', variables={'x': DeclaredVariableSpec(250.0, StateSpec())})
interest = RateInteraction('interest', target='account', state='x', rate=lambda s: 0.05 * s.target.x)
savings = CompositeSystem(elements=[account], interactions=[interest])
simulate(savings, until=1, dt=1.0).values('account.x').round(2).tolist()
```

The metabolic networks of biology build it from their reactions. The shared recipe `NonnegativeElement`, which fixes
one state `x` at zero or above, is on the [page of shared recipes](general.md). The table lists the systems that use
it.

## GroupElement

A `GroupElement` holds many members of one element class as arrays, one row per member. Unlike a single element, it
can sit on the sites of a lattice or the nodes of a network. ...
````

The first block of the sketch should show `[100.0, 60.7, 36.8]` and the second `[250.0, 262.82]`, the values checked
with `CompartmentElement` and the state `n`. The task that writes the page checks them with `DeclaredElement`.

### The order

The change lands after `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`,
`slim-core-laws`, `rename-core-systems`, `move-recipes-to-systems`, and `hoist-constants-and-helper-methods`. Those
changes add, remove, rename, and move the classes this change documents, such as `DeclaredElement`,
`BallNeighborhood`, `ArrayUpdateLaw`, `CompositeSystem`, and the recipes of `skcomplex.systems.general`, and
`slim-core-interactions` already rewrites the docstring of `EncounterInteraction` in part.
`hoist-constants-and-helper-methods` is not yet proposed in `openspec/changes`, so the first task checks that it has
been archived. The evaluate tutorial and the epidemic case study use `identifiability`, so the documents groups also
wait for `add-identifiability-analysis`. The docstring groups do not.

The pilot is the package `components/elements` with its `__init__`, and `CouplingInteraction`, the hardest interaction
docstring. The user approves the style on it before the other packages start. The other packages follow in the order a
reader meets them: the rest of `components`, then `specs` and the root modules, then `analysis` and `experiments`, then
the engine, then `skcomplex.systems.general`. The documents follow the docstrings, so the guide and the generated tables
can read the new summaries, comparisons, and uses. They start with their own pilot: the elements page of the core
components reference with the generated table and fact sheets of elements, the page on how the parts fit, and the build
tutorial, which introduces the running example. The user approves that pilot before the other reference pages,
tutorials, and case studies start. The order after the pilot is the rest of the reference, then the other tutorials in
reading order, then the case studies, then the extras, then the README, CONTRIBUTING, and the gallery.

### Before and after

The "after" texts are written against the names left by the earlier changes. Doctest outputs marked as checked were
run on the current code with the old names. The others are checked when the task runs.

#### 1. The declared element

`NonnegativeElement`, the pilot of the first draft, leaves the core for `skcomplex.systems.general.elements`. Its
rewrite moves with it to task 3.5. The pilot of the core elements is `DeclaredElement`, which stays.

Before:

```python
class DeclaredElement(Element):
    """Element whose states and parameters are data.

    It serves a system that knows them only when it builds the element. The compiler treats a declared variable as it
    treats an annotated argument of the constructor: the spec gives the declaration and the value gives the default.
    ...

    Args:
        ...
        variables:
            The declared states and parameters by name, or `None` for none.
    """
```

After:

```python
class DeclaredElement(Element):
    """An element whose states and parameters are given as data.

    The element takes a mapping from each name to its initial value and its spec, so a system can build it without
    a class of its own. Interactions such as rates, maps, and flows change its states, and the element has no laws of
    its own.

    Unlike `Element`, it needs no subclass, since it takes its states and parameters as data. Choose it when a system
    learns its variables only when it builds them, or when a class for one model would only repeat its arguments.

    It appears in many fields. The number of people infected in an epidemic, the moles of a chemical species in a
    reactor, the money in a bank account, and the tokens in a place of a Petri net are all declared states.

    Args:
        label:
            The name of the element, unique in its system, such as `'infected'` or `'account'`.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        variables:
            The initial value and the spec of each state and parameter, by name, such as a state `x` in `[0, oo)`
            in people or moles, or `None` for none.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        An epidemic compartment and a bank account are the same kind of element.

        >>> from skcomplex.core.components.elements import DeclaredElement
        >>> from skcomplex.core.specs import DeclaredVariableSpec, StateSpec
        >>> infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
        >>> account = DeclaredElement('account', variables={'x': DeclaredVariableSpec(250.0, StateSpec())})

        Interest at 5 percent a year grows the account continuously.

        >>> from skcomplex.core.components.interactions import RateInteraction
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> interest = RateInteraction('interest', target='account', state='x', rate=lambda s: 0.05 * s.target.x)
        >>> system = CompositeSystem(elements=[account], interactions=[interest])
        >>> simulate(system, until=1, dt=1.0).values('account.x').round(2).tolist()
        [250.0, 262.82]
    """
```

The output `[250.0, 262.82]` was checked with `CompartmentElement` and the state `n`. The task checks it again with
`DeclaredElement`.

#### 2. `CouplingInteraction`

Before, the summary and the first of seven paragraphs, about 700 words before `Args`:

```text
Coupling of every member to other members.

The coupling is a reduction over the other members of a function of both members. Every member `i` gets the
reduction over other members `j` of `function(namespace)`, where `namespace.member` reads the states of `i`,
`namespace.other` those of `j`, `namespace.edge` the member of the edge group that joins them, or `None`, and the
namespace also holds the parameters and the time `t`. ...
```

The list of modes is also broken by a paragraph inserted between its second and third items.

After:

```text
An interaction that combines a function of each member and its neighbours.

Every member `i` of a group receives a sum, a mean, a maximum, or another reduction of `function` over the members
`j` it is coupled to. By default the function is the difference `x_j - x_i` of one state. The result adds to the rate
of a state, sets its next value at every step, or steers agents, as `form` says.

Unlike `PairInteraction`, which runs its laws for every pair within a cutoff, it reduces over the members each member is coupled to, by
edges, a group, or hyperedges, and it may set a rate, a map, or an agent rule. Choose it when every member responds to
an aggregate of its neighbours.

It appears in many fields. Phase oscillators, such as fireflies that flash in step or generators of a power grid, sum
the sine of their phase differences. Heat flow between the cells of a lattice and consensus in a social network sum
differences of a state. Flocks of birds and schools of fish average the headings of their neighbours within a radius.

The members `j` are chosen by `mode`:

- `'edges'`: the neighbours of `i` along the edges of its lattice or network.
- `'group'`: every other member of the group, or the members of another group.
- `'next'`: the next member in order, the first after the last, as in a ring.
- `'hyperedges'`: the other members of each hyperedge, a link that joins more than two members.

Args:
    ...
    mode:
        How the members `j` of each member `i` are chosen, one of the modes above. In the mode `'group'` the target
        may also be a list of separate elements, which are then coupled to each other.
    ...

Examples:
    Two phase oscillators, as in the Kuramoto model of synchrony, pull each other by the sine of their difference.

    >>> ...  (the current example, unchanged)
```

The details of the weights, delays, and the lattice form of a rate move from the body to the entries of `weights`,
`delay`, and `form`. The agent radii leave the core with `ZoneCouplingInteraction`, as `move-recipes-to-systems`
decides, so the rewrite drops them from the docstring.

#### 3. `BilinearLaw`, the primitive under encounters

`EncounterInteraction` leaves the core for `skcomplex.systems.general.kinetics`. Its rewrite moves with it to task 3.5,
and its section to `core_components/general.md`. The core keeps `BilinearLaw`, which the encounter passes lower.

Before:

```text
Gain or loss of a target density at a signed rate of a product of densities.

The product is of two densities. The amount of the first density `f` in class `h` that meets the amount of the
second density `g` in class `k` adds to class `i` of the target at the rate `r[h, k]` with probability `B[h, k, i]`,
so the law adds `sum_hk r[h, k] B[h, k, i] f_h g_k` to the rate of the target at `i`. ...
```

After:

```text
A law where members of two distributions meet and change class.

A distribution is a group of elements, one per class of a trait, that holds how much of a population sits in each
class. When an amount of class `h` meets an amount of class `k`, at the rate `rate[h, k]`, it adds to class `i` of
the target with the probability `kernel[h, k, i]`. A conserved law takes the same amount from class `h`, so the total
stays the same.

Unlike `RateLaw`, which gives the rate of one state by an expression, it sums a product of two distributions over
every pair of classes. Choose it when the change of a population comes from binary meetings of its members.

It appears in many fields. Traders exchange wealth when they meet, people shift their opinion after a conversation,
immune cells destroy tumour cells they meet, and gas molecules change velocity when they collide.
```

#### 4. `ArrayUpdateLaw`, which renames `DriveLaw`

Before:

```text
Change of a whole lattice state at once by a function of its array.

An example is adding 1 at a random site. The function receives the state as an array of the lattice shape, or of one
row per member of a group on the edges or hyperedges of a network, with the components of a state of several
components per member, such as a matrix group state, on a last axis. It also receives keyword arguments ...
```

After:

```text
A law that replaces a whole array state by the output of a function.

At every step the function receives the state of every member as one array, shaped like the lattice or with one row
per edge of a network, and returns the new array. It may also receive the random generator, the time, the space, and
the parameters of the element.

Unlike `UpdateLaw`, which gives the next value of each member by an expression, it hands the whole array to a Python
function. Choose it when the change depends on the whole array at once, such as one random site.

It appears in many fields. A sandpile adds a grain at a random site, a molecular model replays recorded
configurations, and a genetic model mutates one random site of a genome.

Examples:
    A sandpile drops one grain on every site.

    >>> from sage.all__sagemath_symbolics import var
    >>> from skcomplex.core.components.laws import ArrayUpdateLaw
    >>> ArrayUpdateLaw(var('height'), lambda values, *, random: values + 1).prepare().targets_
    (height,)
```

#### 5. `PhaseSpace`

Before:

```text
Product of a position lattice and class axes.

It is a lattice whose sites are the points of the product. The position axes come first and sites are in row-major
order, so the sites at one position are contiguous. A class axis has its own cells ...
```

After:

```text
A space of positions combined with extra axes, such as velocity.

Each site is one position paired with one value on every extra axis, called a class axis. A class axis is a range
cut into equal cells, such as velocities from -6 to 6. The sites of one position come one after the other.

It appears in many fields. Kinetic theory follows gas molecules over position and velocity, bacteria are counted
over position and swimming direction, and cell populations over position and age.

Args:
    position:
        The lattice of positions, such as `LatticeSpace((64,), spacing=0.2)` for 64 points 0.2 apart.

    axes:
        The class axes, each an `AxisSpec` with a name, a number of cells, and its lower and upper bounds.

    limit:
        The largest number of sites allowed, which guards memory.
```

#### 6. `recurrence`, an analysis function

Before:

```text
Return the recurrence matrix of the points of a source and its quantification.

Every row of the series of a source is a point, and every series a coordinate, so an embedded series or the
states of a trajectory give a point set. Two points recur when they lie within the radius of each other.
```

After:

```text
Return which states of a run come close to each other, with statistics.

A run recurs when it returns close to a state it visited before. The recurrence matrix marks every pair of times
whose states lie within `radius` of each other. The statistics of its diagonal and vertical lines tell periodic,
chaotic, and random runs apart.

It appears in many fields. It finds regular rhythms in heart-rate records, regime changes in climate series, and
periodic orbits in simulated oscillators.

Args:
    ...
    theiler:
        The number of diagonals next to the main one to leave out, so that states close in time do not count as
        returns. 1 leaves out the main diagonal alone.
```

#### 7. `basin_stability`, an experiment function

Before:

```text
Return the fractions of random initial states that end in each attractor.

The initial states are drawn uniformly within bounds. Every initial state is run to `until` and its trajectory is
labelled by the classifier.
```

After:

```text
Return the share of random initial states that end in each final behaviour.

An attractor is the state or the cycle that a run settles into. The function draws initial states uniformly within
`bounds`, runs each one to `until`, and asks `classifier` which attractor the run ended in. The shares measure how
stable each attractor is against large disturbances.

It appears in many fields. It measures how often a power grid stays in step after a fault, how often a population
survives rather than dies out, and how often a genetic switch ends on rather than off.

Args:
    bounds:
        The range of the initial value of each state, by its dotted path, such as `{'population.x': (0.1, 10)}`.
```

#### Module docstrings

| Module | Before | After |
| --- | --- | --- |
| `elements/_declared.py` | "Define declared elements." and "A declared element takes its states and parameters as data at construction." | "Define the element whose states and parameters are given as data." |
| `laws/_bilinear.py` | "Define bilinear laws." and "They are gains and losses of a density at the rate of the product of two densities." | "Define the law where members of two distributions meet." |
| `systems/general/kinetics/_encounter.py` | "Define encounters." and "Encounters are bilinear moves of a density over activity classes by meetings with another density." | "Define the interaction where members of two distributions meet and change class." |
| `elements/__init__.py` | "Elements of systems." and "Elements are entities that carry states, and groups of identical elements." | "Define elements, the parts of a system that hold its states." and "An element holds named numbers, such as an amount or a position. A group holds many elements of one kind as arrays." |

#### A user guide passage

Before, from `kinetic_systems_of_active_particles.md`:

```text
Describe a population by how its members spread over an activity, such as wealth, an opinion, or the progression of a
tumour cell, and let encounters between members reshape that spread. This is the kinetic theory of active particles of
Bellomo et al. Encounters change the distribution, and its moments give the macroscopic quantities.

## Distributions as site groups

Hold a distribution over nine wealth classes as a group with one member per class:

(a block that defines its own WealthDensityElement)

A distribution over `n` activity values is a group on the sites of a lattice of `n` sites, one member per activity,
`GroupElement(label, element=E, size=n, on='sites')`. The element declares one real state, the density at the member's
activity. The activity values are an array that the laws and the moments receive. No distribution component is needed,
and a population of several functional subsystems is several such groups.
```

After:

````text
A kinetic model describes a population by how its members spread over the values of a trait, such as wealth, an
opinion, or the stage of a tumour cell. Encounters between pairs of members change that spread. The approach follows
the kinetic theory of active particles of Bellomo et al.

## Distributions as groups

A distribution says what share of a population has each value of a trait. Hold it as a group of elements with one
share each, one member per value. This group splits a population over nine wealth classes.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.specs import StateSpec


class ShareElement(Element):
    def __init__(self, label, visual=None, *, x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0):
        super().__init__(label, visual=visual)
        self.x = x


classes = np.linspace(0.0, 1.0, 9)
profile = np.exp(-((classes - 0.5) ** 2) / 0.05)
wealth = GroupElement('wealth', element=ShareElement, size=9, on='sites', values={'x': profile / profile.sum()})
wealth.values['x'].round(3).tolist()
```

The shared recipe `NonnegativeElement` of `skcomplex.systems.general.elements` is the same element, and the page of
shared recipes builds the distribution with it.

Each member holds the share `x` of the population in its class. The class values are a plain array, which the laws
and the moments read. The moments of the distribution, such as its mean, give the quantities of the whole population.
A population with several kinds of members, such as traders and banks, is several such groups.
````

### Dead code

The user asked to scan for dead code and remove it before the change is synced, so a task group runs after the other
documents and before integration. Dead means unreachable from the public API. vulture, run at high confidence from a
throwaway environment and never added as a dependency, and an AST check of private names that nothing in `src`,
`tests`, or `docs` references give the candidates. Every hit is reviewed by hand, because names in `__all__` are API
and some names are used dynamically, through `getattr`, Sage hooks, scikit-learn and Mesa protocols, pickling,
registries, or Numba kernels found as globals. The code goes with the tests that only exercise it, one commit per
package, and behaviour stays identical.

## Risks / Trade-offs

- [The earlier changes rename classes again or slip] → The first task checks that the five changes are archived and
  reads their final names. The rules do not depend on names.
- [Cross-domain uses invented for a part that one field alone uses] → Such a part should not be in the core. The task
  notes it for a later change and writes the one honest use, instead of inventing others.
- [Rewritten examples change their outputs] → Every task runs the doctests of its package, and every example output
  is taken from a run, never written by hand.
- [The `REWRITTEN` constant is scaffolding] → It lives in the test module, shrinks the guard to the rewritten
  packages only while the work runs, and is removed by the last group.
- [The summary rule "A" or "An" reads awkwardly for some classes, such as results and errors] → "A result of
  `recurrence`." and "An error for a part that runs before it is prepared." keep the rule. The pilot shows whether the
  user accepts it.
- [The guide pages repeat the docstrings] → The guide holds the family story and the examples. What a class adds, its
  uses, and its catalog links live in the docstring and the generated table, and the guide links to them.
- [A guide example needs a field realization to look natural] → The example uses the core class, and the prose names
  the realization after it, so the core stays the subject of the page.
- [An example of the first draft used a class that `move-recipes-to-systems` moves or merges] → It is rebuilt from
  parts that stay, such as `DeclaredElement`, `JumpInteraction`, `PairInteraction` with a potential,
  `RadiusNeighborhood` with an angle, and `CompositeSystem`, or it moves to `core_components/general.md`. The coverage
  test rejects an import of `skcomplex.systems` on a family page.
- [Catalog links found through module globals miss a system that builds a class indirectly] → The links are a guide
  to examples, not a complete index. The generator also follows the field `common` packages a system imports.
- [Tutorials and case studies slow the test session, since calibration and sweeps run many simulations] → Small
  budgets, small lattices, and fixed seeds. Each page must run in under a minute, and the task that writes it measures
  this.
- [Calibration and sampling outputs vary across platforms] → Pages print rounded values, or a comparison such as
  `bool(abs(estimate - 0.5) < 0.1)`, never raw floating point from a sampler.
- [The banned phrase list rejects a valid term, such as robust control] → Inline code and titles of cited references
  are exempt, and a sentence can name the concept another way. A term that cannot be rephrased is a reason to change
  the list in review, not to suppress the test.
- [Merging `concepts.md` and `choosing_compatible_components.md` breaks links from elsewhere] → The strict build
  fails on each broken link, and the task updates them.
- [A large diff is hard to review] → One commit per package, and the pilot comes first.

## Migration Plan

Docstrings and documents only. Each task group is one commit on `development`. Rollback is a revert of that commit.
