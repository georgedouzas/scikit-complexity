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
- A pilot the user approves before the bulk of the work.
- A light guard for the rules that a machine can check.

**Non-Goals:**

- No change of a name, a signature, or behaviour. A docstring that reveals a confusing name is noted for a later
  change, not renamed here.
- No style linter, no new tool, and no change to the pydoclint, interrogate, or ruff settings.
- Private functions and methods keep their one-line summaries and are only checked for length and semicolons.
- The docstrings of `skcomplex.systems` are out of scope. They follow the same rules when a later change touches them.

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

Alternative considered: a third-party prose linter such as Vale. It needs its own rules and configuration, and the
user asked for a light guard, not a style linter.

### The guard

Four tests are added to `tests/core/components/test_base.py`, which already walks the core and holds `FIELD_TERMS`.
They walk every module of `skcomplex.core`, with `skcomplex.core.specs` and the root modules added to the walk, and
read the docstrings of modules, public classes, public functions, and public methods.

- `test_docstring_summaries_are_short`: the first line is at most 80 characters.
- `test_docstrings_hold_no_semicolon`: no prose line holds `;`. Lines that start with `>>>` or `...` are code and are
  exempt.
- `test_class_summaries_name_their_kind`: the summary of a public class starts with "A " or "An ", and contains the
  kind word of its name suffix. The map of suffix to word is `Element: element`, `Interaction: interaction`,
  `Law: law`, `Space: space`, `Time: time`, `Neighborhood: neighbourhood`, `System: system`, `Solver: solver`,
  `Backend: backend`, `Observer: observer`, `Spec: spec`, `Result: result`, `Pass: pass`, `Error: error`. A class
  whose suffix is not in the map is checked for the article only. No summary of a public class or function holds a
  word of `FIELD_TERMS`, compared case-insensitively.
- `test_docstrings_say_main_public_methods`: no docstring holds the word "verb" or "verbs".
- `test_components_compare_and_name_uses`: the docstring of every public class of the core component families holds
  a paragraph that starts with "Unlike" and holds "Choose it when", and a paragraph that starts with "It appears in".

The first two hold for the whole core once the few current findings are fixed in the first task group. The other
three hold only after a package is rewritten. So they read a module constant `REWRITTEN`, the tuple of package prefixes
already rewritten, which each task group extends. The last group removes the constant and checks the whole core.

`tests/test_sources.py` gains `test_documents_hold_no_semicolon`, which reads the README and the pages of the user
guide, skips fenced code blocks, and fails on a prose line with `;`.

`tests/test_sources.py` also gains `test_core_components_pages_cover_every_class`. It reads the `__all__` of each
family package and fails when a public class, other than an abstract base, has no `## ClassName` heading on the page
of its family.

Alternative considered: one guard test module under `tests/`. The constitution asks the test tree to mirror the
source tree, and `test_base.py` already holds the cross-cutting checks of the core, so the guard joins it.

### The core components guide

The documents should describe the core components, how they differ, what they offer, and how they support systems in
any field. Two places exist today. The guide under `docs/overview/user_guide` is hand-written, and
`pytest-markdown-docs` runs its blocks in the test session. The parts pages under `parts/` are written by
`docs/generate_parts.py` at build time, with tag tables and examples that `markdown-exec` runs in the documentation
build. Each fact goes to one place.

- **Hand-written explanations in the guide.** A new section `docs/overview/user_guide/core_components/` holds
  `index.md` and one page per family: `elements.md`, `interactions.md`, `laws.md`, `spaces.md` with neighbourhoods
  and times, and `systems.md`. It sits in the navigation after "System classes" and before "Build a custom system".
  Each page opens with what the family is for and how the families fit: an element holds states, an interaction
  emits laws that change them, a law is one equation or rule, a space places members, and a system composes them.
  Then each public class has a `## ClassName` section with one plain sentence of what it is, a link to its row of
  the comparison table, and runnable examples from at least two fields that use the class unchanged. The section
  ends with the catalog systems that use the class, through a link to the generated table, not a hand-written list.
- **Generated tables in the parts pages.** `docs/generate_parts.py` adds a comparison table to the pages of elements,
  interactions, laws, and systems, and a new page `parts/spaces.md` for spaces, neighbourhoods, and times. The columns
  are the class, what it adds and when to choose it, taken from the "Unlike" paragraph, its typical uses, taken from
  the "It appears in" paragraph, and the catalog systems that use it. The tag tables and the compatibility examples
  stay as they are.
- **Catalog links.** A system uses a class when its module, or a module of its field `common` package that it
  imports, defines a subclass of the class or names the class in its globals. The generator finds them by walking
  the registered systems of `skcomplex.systems` and links each to its system page. Subclasses are shown as field
  realizations, such as `CompartmentElement` of biology for `NonnegativeElement`. The links are built at build time,
  so no count goes stale.

The docstring is then the one source of what a class is, what it adds, when to choose it, and where it is used. The
guide adds what a docstring cannot hold: the story of the family and examples from several fields side by side.

Alternatives considered: replace the parts pages with hand-written pages. That copies the tags and the summaries by
hand, and they go stale. Or extend only the generated pages. Generated text cannot tell the story of a family, and
`markdown-exec` blocks do not run in the test session.

The examples of the guide pages use the core alone. Where a field realization exists, the page names it after the
core example, as the catalog uses it. Planned pairs, among others:

| Class | Example in one field | Example in another field |
| --- | --- | --- |
| `NonnegativeElement` with `RateInteraction` | an epidemic, infected people recover at a rate | a bank account, interest grows the balance |
| `PointElement` | a planet under gravity | a pedestrian walking to an exit |
| `GroupElement` | a herd of grazing animals | a portfolio of assets |
| `CouplingInteraction` | fireflies that flash in step | consensus of opinions on a network |
| `BirthDeathInteraction` | births and deaths of a population | firms that enter and leave a market |
| `ArrayUpdateLaw` | grains added to a sandpile | a random mutation of a genome |
| `BallNeighborhood` | toppling sites of a sandpile | the neighbours of a voter model |
| `NetworkSpace` | an epidemic on a contact network | a power grid of coupled generators |
| `MassActionSystem` | a chemical reaction network | an SIR epidemic written as reactions |

#### A sketch of the elements page

````text
# Elements

An element holds the states of a system, the numbers that change as the system runs. An interaction reads these
states and emits laws that change them. A space places the members of a group, and a system composes elements,
interactions, and a space into one model. The table on the [elements parts page](../../../parts/elements.md) compares
every element class.

## NonnegativeElement

A `NonnegativeElement` holds one amount that cannot fall below zero. The same element models people in an epidemic
and money in a bank account.

```python
from skcomplex.core.components.elements import NonnegativeElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

infected = NonnegativeElement('infected', x=100)
recovery = RateInteraction('recovery', target='infected', state='x', rate=lambda s: -0.1 * s.target.x)
epidemic = System(elements=[infected], interactions=[recovery])
simulate(epidemic, until=10, dt=5.0).values('infected.x').round(1).tolist()
```

```python continuation
account = NonnegativeElement('account', x=250.0)
interest = RateInteraction('interest', target='account', state='x', rate=lambda s: 0.05 * s.target.x)
savings = System(elements=[account], interactions=[interest])
simulate(savings, until=1, dt=1.0).values('account.x').round(2).tolist()
```

Biology names it `CompartmentElement`, and physics `PhaseSpaceDensityElement`. The table lists the systems that use
it.

## PointElement

A `PointElement` is a point that moves in a continuous space, with a position and a velocity. Unlike a nonnegative
element, its state is a vector, and the laws that act on it give its acceleration. ...
````

The first block of the sketch shows `[100.0, 60.7, 36.8]` and the second `[250.0, 262.82]`, both checked with
`CompartmentElement` and the state `n`.

### The order

The change lands after `add-system-package`, `slim-core-elements-and-spaces`, `slim-core-interactions`,
`slim-core-laws`, and `hoist-constants-and-helper-methods`. Those changes add, remove, and rename the classes this
change documents, such as `NonnegativeElement`, `BallNeighborhood`, `ArrayUpdateLaw`, and `BirthDeathInteraction`, and
`slim-core-interactions` already rewrites the docstring of `EncounterInteraction` in part.
`hoist-constants-and-helper-methods` is not yet proposed in `openspec/changes`, so the first task checks that it has
been archived.

The pilot is the package `components/elements` with its `__init__`, and `CouplingInteraction`, the hardest
interaction docstring. The user approves the style on it before the other packages start. The other packages follow
in the order a reader meets them: the rest of `components`, then `specs` and the root modules, then `analysis` and
`experiments`, then the engine. The documents follow the docstrings, so the guide and the generated tables can read
the new summaries, comparisons, and uses. They start with their own pilot: the elements page of the core components
guide with the generated comparison table of elements, and the two existing guide pages rewritten in the examples
below. The user approves that pilot before the other family pages and documents start.

### Before and after

The "after" texts are written against the names left by the earlier changes. Doctest outputs marked as checked were
run on the current code with the old names. The others are checked when the task runs.

#### 1. The nonnegative element, which replaces `CompartmentElement` and `DensityElement`

Before:

```python
class CompartmentElement(Element):
    """Compartment, the amount of units in one named state, as a count or a fraction.

    The label of the element names the compartment, such as `'A'`, `'B'`, or `'source'`. A compartment has no laws
    of its own, and reactions and flows between compartments move its units.

    Args:
        ...
        n:
            The initial amount, a nonnegative count or fraction.
    """
```

After:

```python
class NonnegativeElement(Element):
    """An element that holds one nonnegative number.

    The element holds an amount `x` that cannot fall below zero. Interactions such as rates, reactions, and flows
    change it, and the element has no laws of its own.

    Unlike `Element`, it needs no subclass and keeps its one state at zero or above. Choose it when a system tracks
    an amount that cannot be negative, such as a count, a mass, or a sum of money.

    It appears in many fields. The number of people infected in an epidemic, the moles of a chemical species in a
    reactor, the money in a bank account, and the tokens in a place of a Petri net are all nonnegative elements.

    Args:
        label:
            The name of the element, unique in its system, such as `'infected'` or `'account'`.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial amount, a number in `[0, oo)`, in the units of the system, such as people or moles.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        An epidemic compartment and a bank account are the same kind of element.

        >>> from skcomplex.core.components.elements import NonnegativeElement
        >>> infected = NonnegativeElement('infected', x=10)
        >>> account = NonnegativeElement('account', x=250.0)

        Interest at 5 percent a year grows the account continuously.

        >>> from skcomplex.core.components.interactions import RateInteraction
        >>> from skcomplex.core.components.systems import System
        >>> from skcomplex.core.engine.runtime import simulate
        >>> interest = RateInteraction('interest', target='account', state='x', rate=lambda s: 0.05 * s.target.x)
        >>> system = System(elements=[account], interactions=[interest])
        >>> simulate(system, until=1, dt=1.0).values('account.x').round(2).tolist()
        [250.0, 262.82]
    """
```

The output `[250.0, 262.82]` was checked with `CompartmentElement` and the state `n`.

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

The details of the weights, delays, the lattice form of a rate, and the agent radii move from the body to the
entries of `weights`, `delay`, `form`, and `radii`.

#### 3. `EncounterInteraction`

Before:

```text
Encounters of a candidate density with a field density over activity classes.

The encounters move, remove, or create. The candidate and the field are groups of `DensityElement` on the sites of
lattices of activity classes, one member per class. A candidate at class `h` meets a field at class `k` at the rate
`eta[h, k]`. ...
```

After:

```text
An interaction where members of two distributions meet and change class.

A distribution is a group of nonnegative elements, one per class of a trait, that holds how much of a population
sits in each class. When a member of class `h` meets a member of class `k`, at the rate `rate[h, k]`, the first
member moves to class `i` with the probability `kernel[h, k, i]`, is removed, or creates a new member, as `outcome`
says.

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
| `elements/_nonnegative.py` (was `_compartment.py`) | "Define compartments, elements that hold the amount of units in one named state." | "Define the element that holds one nonnegative number." |
| `interactions/_encounter.py` | "Define encounters." and "Encounters are bilinear moves of a density over activity classes by meetings with another density." | "Define the interaction where members of two distributions meet." |
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

A distribution says what share of a population has each value of a trait. Hold it as a group of nonnegative elements,
one member per value. This group splits a population over nine wealth classes.

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement, NonnegativeElement

classes = np.linspace(0.0, 1.0, 9)
profile = np.exp(-((classes - 0.5) ** 2) / 0.05)
wealth = GroupElement('wealth', element=NonnegativeElement, size=9, on='sites', values={'x': profile / profile.sum()})
wealth.values['x'].round(3).tolist()
```

Each member holds the share `x` of the population in its class. The class values are a plain array, which the laws
and the moments read. The moments of the distribution, such as its mean, give the quantities of the whole population.
A population with several kinds of members, such as traders and banks, is several such groups.
````

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
- [Catalog links found through module globals miss a system that builds a class indirectly] → The links are a guide
  to examples, not a complete index. The generator also follows the field `common` packages a system imports.
- [A large diff is hard to review] → One commit per package, and the pilot comes first.

## Migration Plan

Docstrings and documents only. Each task group is one commit on `development`. Rollback is a revert of that commit.
