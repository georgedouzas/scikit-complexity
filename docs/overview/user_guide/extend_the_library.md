# Extend the library

Add your own elements, interactions, laws, and other parts without changing the library, and see where everything
lives.

## Extending the general classes

Give a general class of the library the words of your field, without losing its checks and laws. A general class names
a mathematical operation, and a subclass names what it means in a field. The subclass overrides hooks, and the checks
and the laws of its parent keep working.

### Roles and parameters

Rename a role or a parameter of a general interaction, or fix a parameter to a value.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction, SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class StockElement(Element):
    def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.level = level


class InflowInteraction(RateInteraction):
    def __init__(self, label, *, stock: Annotated[str, RoleSpec()]):
        super().__init__(label, target=stock, state='level', rate=lambda s: 1.0)
        self.stock = stock

    def _get_role(self, role):
        return {'target': 'stock'}.get(role, role)


inflow = CompositeSystem(elements=[StockElement('tank')], interactions=[InflowInteraction('fill', stock='tank')])
round(float(simulate(inflow, until=1, dt=0.1).get_values('tank.level')[-1]), 3)
```

`_get_role` renames a role, here `target` to `stock`, in the symbolic form and the agent form alike. `_get_parameter`
renames a parameter in the same way. An argument that the class does not declare as a parameter is a plain value, so a
subclass fixes a parameter by setting it, as `InflowInteraction` fixes the rate to 1.

```python continuation
class SpinElement(Element):
    def __init__(self, label, visual=None, *, s: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.s = s


class ThermalSamplingInteraction(SamplingInteraction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], beta: Annotated[float, ParameterSpec()] = 1.0):
        energy = lambda s: -s.target.s * neighbors(s.target.other.s)
        super().__init__(label, target=target, state='s', objective=energy)
        self.beta = beta

    def _get_parameter(self, name):
        return {'intensity': 'beta'}.get(name, name)


spins = GroupElement('spins', element=SpinElement, space=LatticeSpace((8, 8)), on='sites')
magnet = CompositeSystem(
    elements=[spins],
    interactions=[ThermalSamplingInteraction('thermal', target='spins')],
    time=DiscreteTime(),
)
abs(simulate(magnet, until=20, random_state=0).states[-1].mean()) > 0.9
```

Here the intensity of `SamplingInteraction` becomes the inverse temperature `beta` of a magnet.

### Variants

Choose the variant of a general interaction with a named value, a spec, or a function. Each argument takes some of
these:

- The pairing of a transfer or a grouping takes a `PartnerSpec` or a function.
- The groups of a grouping, the action of a topology, and the mode of a coupling or a map take their named values or
  a spec of a subclass.
- The move of a sampling takes its named values or a function. It takes no spec, because a function already says what
  a spec would. A spec given to it raises an error that lists what the argument takes.

Every other value raises the same error, so the checks of the parent keep working.

### Partners and settlement

Say whom an agent pairs with in a transfer, and how both partners settle the amount.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import PartnerSpec, StateSpec


class EaterElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        level: Annotated[float, StateSpec()] = 0.0,
        sold: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.level = level
        self.sold = sold

    def _get_laws(self, s):
        return [RuleLaw(lambda agent, system: None)]


def settle(source, target, amounts, system):
    source.level += sum(amounts)
    target.sold += 1.0


space = ContinuousSpace(extent=(10.0, 10.0))
eaters = GroupElement('eaters', element=EaterElement, size=3, space=space)
providers = GroupElement('providers', element=EaterElement, size=2, space=space)
meal = TransferInteraction(
    'meal',
    source='eaters',
    target='providers',
    state='level',
    into='sold',
    amount=lambda source, target, system: 2.0,
    pairing=PartnerSpec(1.0, fallback='nearest'),
    form='agent',
    settlement=settle,
)
system = CompositeSystem(elements=[eaters, providers], interactions=[meal], time=DiscreteTime())
trajectory = simulate(system, until=1, random_state=0)
trajectory.get_agents('eaters', 1).values['level'].tolist()
```

`PartnerSpec(candidates, choice, fallback, capacity, score)` says whom an agent pairs with:

- The candidates are `'all'`, `'neighbors'`, `'cell'`, `'local'` for both of these, a distance, or the name of a state
  that holds a distance per agent. So it works in continuous space too.
- The choice is `'uniform'`, `'score'`, `'nearest'`, or `'all'`.
- The fallback `'nearest'` takes the nearest partner when no one is within reach.

A `settlement(source, target, amounts, system)` of a transfer receives both partners and the drawn amounts, so both ends
credit one draw.

### Agent states that hold objects

Give an agent a state that holds any value, such as a tuple of orders, and tell analyses what a state means.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class QueueElement(Element):
    def __init__(self, label, visual=None, *, orders: Annotated[object, StateSpec(domain=None)] = ()):
        super().__init__(label, visual=visual)
        self.orders = orders

    def _get_laws(self, s):
        return [RuleLaw(self._step)]

    def _step(self, agent, system):
        agent.orders = (*agent.orders, int(system.time))[-2:]


group = GroupElement('queues', element=QueueElement, size=2, values={'orders': [(), (7,)]})
queues = CompositeSystem(elements=[group], time=DiscreteTime())
trajectory = simulate(queues, until=1, random_state=0)
trajectory.get_agents('queues', 1).values['orders'].tolist()
```

A state of an agent with `StateSpec(domain=None)` or `StateSpec(domain=object)` holds any hashable value, such as a
tuple, a string, or a frozen set:

- A snapshot keeps one object per agent in a one-dimensional array.
- A composition counts the agents by value.
- A history keeps any value.
- An analysis that needs numbers names the state and refuses it.

Symbolic systems hold numbers only, in the vectors of their solvers.

`StateSpec(role='position')`, `StateSpec(role='direction')`, `StateSpec(role='phase')`, and
`StateSpec(role='velocity')` tell analyses what a state means. Analyses read positions, directions, phases, and
velocities from these roles, not from the names of the states. So a state named `phase` is no phase unless it says so.

- A group is a particle group when a state has the role `'position'`, and the first derivative of that state is its
  velocity.
- The scalar states of any other group count as series, one per member. A displacement refuses them, since they are no
  positions.
- A variable group has no fixed members, so its states are series only when named, with NaN in the rows that hold no
  member.
- In a continuous space, a vector state with the role `'position'` is the place of its element. The boundary acts on it,
  and neighbourhoods measure distances from it.
- `StateSpec(role='site')` does the same for a place that does not make a member a particle, such as the fixed sites of
  oscillators on a ring.

### Checking arguments

Check the arguments of a new part at the start of its `_prepare`. The constructor only stores its arguments, so a part
with a bad argument is built without error, and preparing it, alone or within its system, raises the error. Start
`_prepare` with `super()._prepare(context)`: the base of the parts checks the label and the declared specs first, then
the part checks its own arguments before it sets any attribute that ends in `_`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec


class TankElement(Element):
    def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 1.0, capacity=10.0):
        super().__init__(label, visual=visual)
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

A spec, such as `StateSpec`, is a frozen value, not an estimator: it has neither `prepare` nor `run`. The part that holds
a spec checks it through its private hook `_check` when it reads it, as the base of the parts does for the specs of the
annotated arguments.

## Package layout

Find where every class lives, so you know what to import and where a part of your own belongs. The package has two
parts:

- `skcomplex.core` is the general library: what a system is, how it runs, and how you analyse and experiment with it.
- `skcomplex.systems` is the catalog of ready-made systems, built from the core.

```python
from skcomplex.core.base import BaseConfig
from skcomplex.core.components.elements import Element
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

[issubclass(cls, BaseConfig) for cls in (Element, CompositeSystem, LotkaVolterraSystem)]
issubclass(LotkaVolterraSystem, BaseSystem), LotkaVolterraSystem.field
```

Every estimator, an element, an interaction, a law, a system, and the engine that runs them, derives from one root
class, `BaseConfig`. `simulate`, the analyses, and the experiments, calibration among them, are functions of these
estimators. Every ready-made system
belongs to a field and a subfield of the catalog.

The parts of the core are layered, so that each imports only the parts below it. From top to bottom:

1. `skcomplex.core.experiments` runs many simulations.
2. `skcomplex.core.analysis` measures systems, trajectories, and data.
3. `skcomplex.core.engine` runs systems.
4. `skcomplex.core.components` defines what a system is.
5. `skcomplex.core.specs` holds the specs.
6. `skcomplex.core.base` holds `BaseConfig`.
7. `skcomplex.core.types` and `skcomplex.core.errors` hold the types and the errors.

No part of the core imports `skcomplex.systems`, and the systems import only the public modules of the core. Tests of
the import contracts enforce the layering.

### Where a new part goes

One ordered test, the placement rule, decides where any part goes, now and in the future. `tests/test_sources.py`
checks it mechanically.

1. A structural base, such as a `Base` class, the composite system, or a base where laws run per member, edge,
   hyperedge, or pair, goes to the core.
2. A part that the engine handles by type, through a compile pass, a kernel, a solver, a backend lowering, or an
   analysis that checks the class or its spec, goes to the core, and so does an operator generated from one function.
   It is named by its mathematics and must be cross-domain in principle.
3. Every other part is a recipe, expressible with core parts. Its users place it. Several fields:
   `skcomplex.systems.general.<concept>`. One field with several systems: `systems/<field>/common/<family>`. One system:
   that system's module. A field with its own term names it, and `general` uses neutral names.
4. A recipe that needs something the core cannot express reveals a missing core primitive, which enters by step 2.
   Engine support never lives inside a recipe.
5. The core never imports `skcomplex.systems`.

One example for each step:

1. `HyperedgeInteraction` runs its laws once per hyperedge, and `CompositeSystem` holds the parts of a system, so both
   are structural bases of the core.
2. `GroupingInteraction` stays in the core, since a compile pass lowers its symbolic form by type to kernels over
   random groups redrawn every step. Its name is the mathematics it does, and economics and social both form groups.
3. `MovementInteraction` returns agent rules, which the core already runs, and systems of biology, economics,
   engineering, mathematics, and social use it, so it lives in `skcomplex.systems.general.agents`. The toppling of a
   sandpile serves several systems of physics only, so it lives in a family of the physics `common` package under the
   physics name `TopplingInteraction`. The satisfiability of logical constraints serves one system, so
   `SatisfiabilityInteraction` lives in the module of `PerceptronSatisfiabilitySystem`.
4. The satisfiability once had a check of its own in the compiler, which is engine support inside a recipe. The
   check that a role accepts only Boolean states is a primitive of the core, `RoleSpec(domains=...)`, so the recipe
   declares its role with it and the compiler keeps no code for the class.
5. `sparse_identification` of the core returns a `CompositeSystem` of core parts, never a class of
   `skcomplex.systems`, so the core runs without the catalog.

### The root modules

Every package of the core shares the root modules:

- `skcomplex.core.base` holds `BaseConfig`, the root of every configurable class. It stores the settings of a class
  under their own names, as scikit-learn does, so `get_params`, `set_params`, and `clone` work on every class. It gives
  every class the main public methods `prepare` and `run`, and `get_tags` gives the capability tags of every part.
- `skcomplex.core.errors` holds the errors and warnings, such as `ValidationError`, `RoleError`, `SolverError`, and
  `RewritingError`.
- `skcomplex.core.types` holds the types that every layer reads, such as arrays, functions, keys, and kinds.
- `skcomplex.core.specs` holds the specs, frozen values that declare, record, or hold what the classes read. These
  include the specs that annotate states, parameters, and roles, such as `StateSpec`, `ParameterSpec`, and `RoleSpec`,
  the specs of runs and structures, such as `ObservableSpec` and `WienerProcessSpec`, and the specs of rules, domains,
  and geometries, such as `TableSpec`, `LinearRuleSpec`, `SubshiftSpec`, and `GeometrySpec`.

### The core

`skcomplex.core.components` defines what a system is, and nothing about how it runs. Its modules are listed in their
import order, so a module imports only the modules above it:

- `types`: the types of the components, such as the targets of laws and the pairings of transfers.
- `base`: `BasePart`, the base of every part of a system, built on `BaseConfig`.
- `spaces`: the continuous, lattice, network, mesh, and Cayley spaces, their neighbourhoods, and the kinds of time.
- `laws`: the `BaseLaw` base and every law. `laws.utils` holds what laws are written with besides Sage, such as
  `where`, reductions, draws, delays, and calls. The helpers that only the compiler reads, such as `reductions_in`,
  stay in its private modules, and the engine imports them from there.
- `elements` and `interactions`: `Element`, `GroupElement`, `FieldElement`, and the other elements, and the binding
  bases `Interaction`, `NeighborInteraction`, `PairInteraction`, and `HyperedgeInteraction`, with the general
  interactions built on them.
- `systems`: `BaseSystem` and its `RunSettings`, and the composite `CompositeSystem`.
- `specs`: the specs that name classes of the components, as a compilation reads them, such as `ScenarioSpec`.

### The engine

`skcomplex.core.engine` runs systems. Its packages are listed in their import order too:

- `types`: the types that only the engine reads.
- `problems`: the compiled problems, their layouts of states and parameters, and their numeric functions.
- `backends`: the NumPy, Numba, and JAX backends, which lower expressions to numeric functions.
- `compile`: the compilers and the passes that turn a system into a problem. They are private, except `BaseCompiler`,
  `BasePass`, and `BaseLowering`, through which a system package brings a form of its own with the `compiler` run
  setting.
- `solvers`: the solver contract, `step_count`, and every solver.
- `runtime`: `simulate`, clocks, observers, and trajectories.
- `viz`: the visuals and the Rerun observer.
- `translators`: the translators of system files of other tools, such as `SystemDynamicsTranslator`.

### Analysis and experiments

`skcomplex.core.analysis` holds the analysis functions of systems, trajectories, and data, their frozen results, and
`bind_analysis`, with the types that only they read in `skcomplex.core.analysis.types`. `skcomplex.core.experiments`
holds the experiment functions over designs of parameters, scenarios, and replicates, sensitivity, basin stability, and
orbit diagrams, and calibration. Its `types` and `specs` hold the rows of a design and the runs of an
experiment.

### The systems

`skcomplex.systems` is the catalog. `all_systems` lists its classes, and each class is importable from its subfield
package. The systems are organised by field, `biology`, `chemistry`, `computer_science`,
`economics`, `engineering`, `mathematics`, `physics`, and `social`, and by subfield inside each field, such as
`physics.mechanics` or `biology.ecology`.

- A subfield package is self-contained, and imports no other subfield.
- A realization that several subfields of one field share lives in a family of the `common` package of that field,
  such as `physics.common.avalanches`.
- A realization that several fields share lives in a concept subpackage of `skcomplex.systems.general`, such as its
  `agents` or `reactions`, described in [General realizations](core_components/general.md).
- Every system is written with the same public classes of the core as a custom system, so its source also works as a
  pattern.
