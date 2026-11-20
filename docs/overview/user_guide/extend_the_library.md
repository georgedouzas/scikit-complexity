# Extend the library

This page adds elements, interactions, laws, and other components of your own without changing the library, and shows
where every class lives. A component of your own is a subclass of a class of the core, so the checks and the laws of its
parent keep working.

## Extending the general classes

This section gives a general class of the library the words of your field. A general class names a mathematical
operation, and a subclass names what it means in a field. The subclass overrides hooks, the private methods that a
class reads to let a subclass change one step, and the checks and the laws of its parent keep working.

### Roles and parameters

This section writes an interaction with the roles and parameters of your field. A [role](glossary.md#role) is the
name under which an interaction reads and changes an element, an argument typed `Role`. Every other argument is a
parameter of the relation. So an interaction names its roles and parameters in its own `__init__`, and its `_prepare`
sets its laws from them. The example fills a tank through a role `stock` at the rate `q`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Inflow(Interaction):
    def __init__(self, label, *, stock: Role, q=1.0):
        super().__init__(label)
        self.stock = stock
        self.q = q

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [RateLaw(s.stock.level, s.q)]


inflow = System(elements=[Element('tank', values={'level': 1.0})], interactions=[Inflow('fill', stock='tank')])
round(float(inflow.prepare().simulate(until=1, dt=0.1).get_values('tank.level')[-1]), 3)
```

The level rises from 1.0 to 2.0 in one unit of time. The laws read the role as `s.stock` and the parameter as `s.q`,
so `get_params` and `set_params` name the parameter `fill__q` in the system. A subclass of an interaction of the
catalog renames nothing: it keeps the arguments of its parent and adds its own.

### Variants

This section chooses the variant of a general binding or law. A variant is given as a named value, a spec, or a
function, and each argument takes some of these.

- The binding `along` takes `True`, the name of a layer, `'hyperedges'`, `('hyperedges', k)`, `RandomPartition(k)`, or
  `'neighborhoods'`.
- The `schedule` of a local rule takes `'synchronous'`, `'random-sequential'`, `'random-site'`, `'checkerboard'`, or
  `'colored'`, and a `RadiusNeighborhood` takes the `metric` of its distance, such as `'chebyshev'` or `'manhattan'`.
- The pairing of `pairs` takes a `PartnerSpec` or a function that returns the pairs.

Every other value raises an error that lists what the argument takes, so the checks of the parent keep working.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import System

try:
    flip = Interaction(
        'flip',
        roles={'target': 'cells'},
        laws=lambda s: [UpdateLaw(s.target.x, lambda x: 1 - x, neighborhood=RadiusNeighborhood(1), schedule='often')],
    )
    System(elements=[Element('cells', values={'x': 0}, space=LatticeSpace(5))], interactions=[flip]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message names the unknown schedule `'often'` and lists the five schedules a local rule takes.

### Partners and settlement

This section says whom an agent pairs with, and how both partners settle the amount. A rule of the whole system,
`UpdateLaw(s, rule)`, reads the agents of two groups with `system.get_agents`, pairs them with `pairs` of
`skcomplex.core.components.laws.utils`, and settles every pair. In the example, three eaters buy a meal of 2 units
from one of two providers, each agent a member of a group in a
[`ContinuousSpace`](core_components/spaces.md#continuousspace) of 10 by 10.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import PartnerSpec, UpdateLaw
from skcomplex.core.components.laws.utils import pairs
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class Meal(Interaction):
    def __init__(self, label, *, eaters: Role = 'eaters', providers: Role = 'providers', amount=2.0):
        super().__init__(label)
        self.eaters = eaters
        self.providers = providers
        self.amount = amount

    def _prepare(self, s):
        super()._prepare(s)
        self.laws_ = [UpdateLaw(s, self.serve)]

    def serve(self, system):
        reach = PartnerSpec(1.0, fallback='nearest')
        for eater, provider in pairs(system.get_agents('eaters'), system.get_agents('providers'), system.rng, reach):
            eater.level += self.amount
            provider.sold += 1.0


space = ContinuousSpace(extent=(10.0, 10.0))
eaters = Element('eaters', values={'level': 0.0, 'sold': 0.0}, size=3, space=space)
providers = Element('providers', values={'level': 0.0, 'sold': 0.0}, size=2, space=space)
system = System(elements=[eaters, providers], interactions=[Meal('meal')])
trajectory = system.prepare().simulate(until=1, random_state=0)
trajectory.get_agents('eaters', 1).values['level'].tolist()
```

Each of the three eaters holds 2.0 after one step. `PartnerSpec(candidates, choice, fallback, capacity, score)` says
whom an agent pairs with.

- The candidates are `'all'`, `'neighbors'`, `'cell'`, `'local'` for both of these, a distance, or the name of a value
  that holds a distance per agent. So it works in continuous space too.
- The choice is `'uniform'`, `'score'`, `'nearest'`, or `'all'`.
- The fallback `'nearest'` takes the nearest partner when no one is within reach.

Here the candidates lie within a distance of 1.0, and the fallback pairs an eater with its nearest provider when none
does. The rule settles each pair in plain Python, so both partners change in one place, the eater by the amount and
the provider by one sale.

### Agent states that hold objects

This section gives an agent a value that holds any object, such as a tuple of orders, and tells analyses what a value
means. Each queue below keeps its last two orders, and an `UpdateLaw` whose target is the element and whose value is a
function of one agent and the system, a rule, appends the current time to them at each step.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.systems import System


class QueueElement(Element):
    def __init__(self, label, *, orders: object = (), size=None):
        super().__init__(label, size=size)
        self.orders = orders


def record(agent, system):
    agent.orders = (*agent.orders, int(system.time))[-2:]


starts = np.empty(2, dtype=object)
starts[:] = [(), (7,)]
queues = QueueElement('queues', orders=starts, size=2)
ordering = Interaction('ordering', roles={'target': 'queues'}, laws=lambda s: [UpdateLaw(s.target, record)])
trajectory = System(elements=[queues], interactions=[ordering]).prepare().simulate(until=1, random_state=0)
trajectory.get_agents('queues', 1).values['orders'].tolist()
```

After one step the first queue holds `(1,)` and the second `(7, 1)`. A value typed `object` holds any hashable value,
such as a tuple, a string, or a frozen set, one per agent in an array of objects. The library records such a value as follows.

- A snapshot keeps one object per agent in a one-dimensional array.
- A composition counts the agents by value.
- A history keeps any value.
- An analysis that needs numbers names the value and refuses it.

Symbolic systems hold numbers only, in the vectors of their solvers.

Analyses read what a value means from where it lives and how it is typed, not from its name alone.

- A value named after an axis of the space of its element, such as `x` and `y` on a continuous space, is a position,
  and its first derivative, `x'`, is the velocity.
- A value typed `Annotated[float, Period(2 * pi)]` is an angle, read as a phase, or as a heading when its element has
  a position.
- The scalar values of any other element of many members count as series, one per member. A displacement refuses
  them, since they are no positions.
- An element whose members are added or removed has no fixed members, so its values are series only when named, with
  NaN in the rows that hold no member.
- In a continuous space, the boundary acts on the positions, and neighbourhoods measure distances from them.

### Checking arguments

This section checks the arguments of a new component. The constructor only stores its arguments, so a component with a
bad argument is built without error. Preparing it, alone or within its system, raises the error. The check goes at the
start of the hook `_prepare`, which begins with `super()._prepare(context)`. The base of the components checks the label
and the allowed values first, and then the component checks its own arguments before it sets any attribute that ends
in `_`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.errors import ValidationError


class TankElement(Element):
    def __init__(self, label, level=1.0, capacity=10.0):
        super().__init__(label)
        self.level = level
        self.capacity = capacity

    def _prepare(self, context=None):
        super()._prepare(context)
        if self.capacity <= 0:
            error_msg = f'The capacity of {self.label!r} is {self.capacity}. Give a positive capacity.'
            raise ValidationError(error_msg)
        self.free_ = self.capacity - self.level


tank = TankElement('tank', capacity=-1.0)
try:
    tank.prepare()
except ValidationError as error:
    message = str(error)
message, hasattr(tank, 'free_'), TankElement('tank').prepare().free_
```

The message reads "The capacity of 'tank' is -1.0. Give a positive capacity." The failed tank sets no `free_`, and a
tank with the default capacity of 10 and level of 1 has 9.0 free. The fix is a positive capacity.

A spec, such as `VisualSpec`, is a frozen value, not an estimator, so it has neither `prepare` nor `run`. The component
that holds a spec checks it through the private hook `_check` of the spec when it reads it.

## Package layout

This section shows where every class lives, so you know what to import and where a component of your own belongs. The
package has two subpackages.

- `skcomplex.core` is the general library: what a system is, how it runs, and how you analyse it and experiment with
  it.
- `skcomplex.systems` is the catalog of ready-made systems, built from the core.

```python
from skcomplex.core.base import BaseConfig
from skcomplex.core.components.elements import Element
from skcomplex.core.systems import BaseSystem, System
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

[issubclass(cls, BaseConfig) for cls in (Element, System, LotkaVolterraSystem)]
```

All three checks return `True`. Every estimator, an element, an interaction, a law, a system, and the engine that runs
them, derives from one root class, `BaseConfig`. A system is prepared and simulated by its own methods, and the
analyses and the experiments, calibration among them, are functions of these estimators. Every ready-made system belongs to a field and a subfield of the catalog.

```python continuation
issubclass(LotkaVolterraSystem, BaseSystem), LotkaVolterraSystem.field
```

`LotkaVolterraSystem` is a `BaseSystem` of the field path `'biology/ecology'`. The packages of the core are layered,
so that each imports only the packages below it. From top to bottom:

1. `skcomplex.core.experiments` runs many simulations.
2. `skcomplex.core.analysis` measures systems, trajectories, and data.
3. `skcomplex.core.engine` runs systems.
4. `skcomplex.core.components` defines what a system is, with the specs its classes read.
5. `skcomplex.core.base` holds `BaseConfig` and `BaseSpec`.
6. `skcomplex.core.types`, `skcomplex.core.enums`, and `skcomplex.core.errors` hold the types, the kinds, and the errors.

No part of the core imports `skcomplex.systems`, and the systems import only the public modules of the core. Tests of
the import contracts enforce the layering.

### Where a new component goes

This section decides where a component of your own belongs. One ordered test, the placement rule, decides where any
component goes, and `openspec/scripts/check_conventions.py` checks it mechanically. The page [General
realizations](core_components/general.md#where-a-component-goes) quotes the rule word for word.

1. A structural base, such as a `Base` class, `Element`, `Interaction`, `System`, a space, or a neighbourhood, goes to
   the core.
2. A law kind, an operator generated from one function, such as `laplacian` or `relax`, and numerics the engine handles
   by type, through a compile pass, a kernel, a solver, a backend lowering, or an analysis, go to the core. They are
   named by their mathematics and must be cross-domain in principle. The engine keys only on the seven law kinds and
   never names a class of `skcomplex.systems`.
3. Every other component is a recipe, written with core components, and a law of one field is first written as a
   recipe of the seven law kinds and the core operators. Its users place it. Several fields:
   `skcomplex.systems.general.<concept>`. One field with several systems: `systems/<field>/common/<family>`. One
   system: that system's module. A field with its own term names it, and `general` uses neutral names.
4. A recipe that needs something the core cannot express reveals a missing core primitive, which enters by step 2.
   Numerics that serve one field alone, such as the Lindblad equation or rewriting, come as a solver or a compiler
   built on the public engine bases, in the package of that field, and its elements name it.
5. The core never imports `skcomplex.systems`.

Analyses follow the rule by their users too. An analysis that measures what systems of several fields share goes to
`skcomplex.core.analysis`, and an analysis of one field goes to `systems/<field>/common/<family>`.

One example for each step follows.

1. [`Interaction`](core_components/interactions.md#interaction) binds roles, pairs within a neighbourhood, or the
   edges of a network, and `System` holds the components of a system, so both are structural bases of the core.
2. [`RandomPartition`](core_components/interactions.md#randompartition) stays in the core, since a compile pass
   lowers the binding `along=RandomPartition(k)` by type to kernels over random groups redrawn every step. Its name is
   the mathematics it does, and economics and social science both form groups.
3. [`MovementInteraction`](core_components/general.md#movementinteraction) returns agent rules, which the core already
   runs, and systems of biology, economics, engineering, mathematics, and social science use it, so it lives in
   `skcomplex.systems.general.agents`. The toppling of a sandpile serves several systems of physics only, so it lives
   in a family of the physics `common` package under the physics name `TopplingInteraction`. The satisfiability of
   logical constraints serves one system, so `SatisfiabilityInteraction` lives in the module of
   `PerceptronSatisfiabilitySystem`.
4. The satisfiability once had a check of its own in the compiler, which is engine support inside a recipe. The check
   that a role accepts only Boolean states is a primitive of the core, `Annotated[Role, 'boolean']`, so the recipe
   declares its role with it and the compiler keeps no code for the class.
5. `sparse_identification` of the core returns a `System` of core components, never a class of
   `skcomplex.systems`, so the core runs without the catalog.

### The root modules

This section lists the modules that every package of the core shares.

- `skcomplex.core.base` holds `BaseConfig`, the root of every configurable class. It stores the settings of a class
  under their own names, as scikit-learn does, so `get_params`, `set_params`, and `clone` work on every class. It gives
  every class the main public methods `prepare` and `run`, and `get_tags` returns the capability tags of every
  component.
- `skcomplex.core.errors` holds the errors and warnings, such as `ValidationError`, `RoleError`, `SolverError`, and
  `RewritingError`.
- `skcomplex.core.types` holds the types that every layer reads, such as arrays, functions, and keys.
- `skcomplex.core.enums` holds the kinds that every layer reads, such as `DataKind`, `FormKind`, and `ElementKind`.
- `skcomplex.core.base` holds `BaseSpec`, the base of the specs, frozen values that declare, record, or hold what the
  classes read. Each spec lives beside the class that reads it, such as `VisualSpec` beside the elements and
  `RoleSpec` beside the components.

### The core

This section lists the modules of `skcomplex.core.components`, which defines what a system is and nothing about how it
runs. They are listed in their import order, so a module imports only the modules above it.

- `types`: the types of the components, such as the targets of laws and the pairings of transfers.
- `base`: `BaseComponent`, the base of every component of a system, built on `BaseConfig`.
- `spaces`: the continuous, lattice, network, cell complex, and phase spaces, and their neighbourhoods.
- `laws`: the `BaseLaw` base and every law. `laws.utils` holds what laws are written with besides Sage, such as
  `where`, reductions, draws, delays, and calls. The helpers that only the compiler reads, such as `reductions_in`,
  stay in its private modules, and the engine imports them from there.
- `elements` and `interactions`: `Element`, which covers one member, many members, and one member per site, and
  `Interaction`, which binds roles, pairs within a neighbourhood, or the edges of a network.

`skcomplex.core.systems` holds `BaseSystem`, the base of the ready-made systems, and `System`, the composite of
elements, interactions, and nested systems.
- `specs`: the specs that name classes of the components, as a compilation reads them, such as `ScenarioSpec`.

### The engine

This section lists the packages of `skcomplex.core.engine`, which runs systems. They are listed in their import order
too.

- `types`: the types that only the engine reads.
- `problems`: the compiled problems, their layouts of states and parameters, and their numeric functions.
- `backends`: the NumPy, Numba, and JAX [backends](glossary.md#backend), which lower expressions to numeric functions.
- `compile`: the compilers and the passes that turn a system into a problem. They are private, except `BaseCompiler`,
  `BasePass`, and `BaseLowering`, through which a system package brings a form of its own.
- `solvers`: the solver contract, `step_count`, and every solver.
- `runtime`: clocks, observers, and trajectories.
- `viz`: the visuals and the Rerun observer.
- `translators`: the translators of system files of other tools, such as `SystemDynamicsTranslator`.

### Analysis and experiments

This section names the two packages that study runs. `skcomplex.core.analysis` holds the analysis functions of
systems, trajectories, and data, their frozen results, and `bind_analysis`, with the types that only they read in
`skcomplex.core.analysis.types`. `skcomplex.core.experiments` holds the experiment functions over designs of
parameters, scenarios, and replicates, sensitivity, basin stability, orbit diagrams, and calibration. Its `types` and
`specs` hold the rows of a design and the runs of an experiment.

### The systems

This section describes the layout of the catalog. `all_systems` lists its classes, and each class is importable from
its subfield package. The systems are organised by field, `biology`, `chemistry`, `computer_science`, `economics`,
`engineering`, `mathematics`, `physics`, and `social`, and by subfield inside each field, such as `physics.mechanics`
or `biology.ecology`.

- A subfield package is self-contained, and imports no other subfield.
- A realization that several subfields of one field share lives in a family of the `common` package of that field,
  such as `physics.common.avalanches`.
- A realization that several fields share lives in a concept subpackage of `skcomplex.systems.general`, such as its
  `agents` or `reactions`, described in [General realizations](core_components/general.md).
- Every system is written with the same public classes of the core as a custom system, so its source also serves as a
  pattern.
