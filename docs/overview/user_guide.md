# User guide

This guide follows what you do as a modeler, in order. Start with a ready-made system, change it, then build your own,
simulate it, observe it, analyse it, run experiments over many runs, and extend the library with parts of your own.
Each part builds only on the ones before it.

```python
from skcomplex.systems import create_simulation, describe

describe('sir').formalism
create_simulation('sir').prepare().run().trajectory_.values('I.n').max().round()
```

There are two ways in.

- **Select a ready-made system.** `catalog` lists the systems, `describe` tells you what one is, `create_system`
  returns it, and `create_simulation` returns a simulation of it with its tuned settings. You import these four
  functions from `skcomplex.systems` and nothing else.
- **Build a custom system.** You write elements and interactions with `skcomplex.core.components`, use specs from
  `skcomplex.core.specs`, simulate with `skcomplex.core.engine`, and study the results with `skcomplex.core.analysis`.

Every ready-made system is written with the same classes as a custom one, so the page of a system under
[Systems](../systems/index.md) also works as a pattern. The section
[Choosing compatible components](#choosing-compatible-components) shows how to find which parts of a custom system fit
together.

## Select a ready-made system

Pick a system from the catalog by its name, read what it is, and simulate it with the settings it was tuned for.

### The catalog

Pick a ready-made system by name, read what it is, and simulate it with the settings it was tuned for.
`create_simulation` returns a simulation that is ready to prepare and run.

```python
from skcomplex.systems import create_simulation, describe

entry = describe('sir')
entry.description, entry.parameters, entry.formalism
trajectory = create_simulation('sir').prepare().run().trajectory_
trajectory.values('I.n').max().round()
```

`describe` tells you what a system is: its field, description, tags, references, the parameters of its factory with
their defaults, its default run settings, and the formalism the compiler infers for it. `create_simulation` builds the
system and wraps it in a simulation that holds the system as `system`, the default end time as `until`, and the tuned
time step, solver, observables, initial state, and seed `random_state=0`. Every object has the same two verbs, so you
call `prepare` and then `run`.

You import four functions from `skcomplex.systems` and nothing else.

- `catalog` lists the systems, filtered by a field, a field path, or a tag.
- `describe` gives one system by name.
- `create_system` builds the system by name and returns it.
- `create_simulation` returns a simulation of it with its default run settings.

#### The system catalog

Every ready-made system is a factory registered with its field path, description, tags, references, and default run
settings. `catalog` filters by a field, such as `biology`, by a field path, such as `biology/ecology`, or by a tag.

```python
from skcomplex.systems import catalog, create_system, describe

[entry.name for entry in catalog(field='biology/ecology')]
sorted({entry.field for entry in catalog(field='chemistry')})
[entry.name for entry in catalog(tag='oscillation')]
describe('lotka_volterra').formalism
predation = create_system('lotka_volterra', a=0.2)
predation
```

The catalog belongs to the catalog alone. The engine and the experiments take a system or a factory, never a system
name. So you create a system of the library with `create_system`, or you pass its factory.

Each system has a field path `<field>/<subfield>`, which `catalog` takes as a filter.

| Field | Subfields |
| --- | --- |
| `physics` | `mechanics`, `celestial_mechanics`, `nonlinear_oscillators`, `synchronization`, `statistical_mechanics`, `criticality`, `soft_matter`, `active_matter`, `fluids`, `waves`, `pattern_formation`, `heat`, `cosmology`, `climate`, `oceanography`, `seismology` |
| `chemistry` | `kinetics`, `pattern_formation` |
| `biology` | `ecology`, `epidemiology`, `evolution`, `neuroscience`, `systems_biology`, `physiology`, `collective_behaviour`, `oncology` |
| `economics` | `macroeconomics`, `markets`, `finance`, `econophysics`, `resources`, `innovation` |
| `social` | `opinion_dynamics`, `segregation`, `cooperation`, `spreading`, `crowds`, `organizations`, `language` |
| `engineering` | `traffic`, `circuits`, `control`, `power_systems`, `robotics`, `processes`, `mechatronics` |
| `mathematics` | `dynamical_systems`, `stochastic_processes`, `networks`, `fractals`, `fractional_dynamics` |
| `computer_science` | `automata`, `evolutionary_computation`, `neural_networks` |

A system lives where the science that poses it lives, and its tags carry its reach into other fields. The library keeps
the catalog apart from the core by three rules, which the checks of the package layout enforce.

- The core never imports the systems.
- Every system is built only from the public names of the core.
- Every subfield package is self-contained, so no system package imports another. Code that several subfields of one
  field share lives in the `common` package of that field, and no field uses the `common` package of another.

#### Optional extras

Most systems run on the default install. A few take a library from an optional extra. Without the extra they stay in
the catalog, and building one raises `MissingExtraError` with the name of the system, the extra, and the install
command.

| Extra | Library | Systems | Install notes |
| --- | --- | --- | --- |
| `structures` | Biopython | `fuzzy_oil_drop`, `early_stage_folding_intermediate` | Biopython 1.88 publishes no wheel for macOS on Intel, so the extra builds it from source there with a C compiler. |
| `rna` | ViennaRNA | `rna_neutral_networks` | The licence of ViennaRNA allows research and commercial use, forbids redistribution for a fee, and is not OSI approved, so the default install never brings it. |
| `trajectories` | MDTraj | none, it reads molecular dynamics trajectories for the systems biology systems | MDTraj is under the LGPL. |

#### Systems as patterns

The ready-made systems are examples of the core interface, not a separate one. So each is also the pattern for a system
of your own. The page of every field, under [Systems](../systems/index.md) in the documentation, shows for each system:

- its factory,
- its element and interaction types, with their states, parameters, and laws as written,
- for differential equations and maps, its equations as `SymbolicAnalysis('equations')` returns them.

A system of your own follows the same steps. Declare the element and interaction types, and write a function that
returns the system. The section [Choosing compatible components](#choosing-compatible-components) shows how to find
which parts fit together, and the
[contributing guide](../development/contributing.md#adding-a-system-to-the-catalog) covers adding a system to the
catalog.

#### Inventory

The system inventory, `docs/overview/inventory.toml`, records every system surveyed from the reference books and
articles, with its sources and its status: implemented, pending a planned change, or blocked by a missing feature of the
core. A system the core cannot express is recorded as blocked on the OpenSpec change that specifies the missing general
feature. It is never worked around inside the system package. A test checks the inventory against the catalog and the
changes. Every registered system has its own page under Systems in the documentation, by field and subfield.

Systems exist first to exercise the core, and ready-made systems are a by-product. Every entry lists the core primitives
it exercises, such as `rates`, `jumps`, `groups`, `reductions`, `member-offsets`, `lattices`, `fields`, `agents`, or
`stocks-and-flows`, from a vocabulary the inventory defines. A primitive that the core does not have yet names the
change that adds it. An entry's priority is `core` when it exercises a primitive, or a combination of primitives, that
no earlier implemented entry exercises, or when it needs a missing one. It is `example` when it repeats a covered
pattern. So new systems go first where the core is least exercised.

## Change it

Give a ready-made system your own arguments and run settings, or nest it inside a larger system.

### Changing a ready-made system

A ready-made system changes in three places. Its factory arguments change the system itself, its run settings change
how the simulation runs, and its `system` can leave the catalog and run under a simulation of your own.

#### Factory arguments

Every factory argument has a default, and `describe` lists them. Pass new values by name to `create_simulation` or to
`create_system`.

```python
from skcomplex.systems import create_simulation, create_system, describe

describe('lotka_volterra').parameters
simulation = create_simulation('lotka_volterra', r=1.5, predators=5.0)
simulation.prepare().run().trajectory_.values('prey.n').max().round()
logistic = create_system('lotka_volterra', K=200.0)
logistic.interactions
```

An argument can change more than a number. With a finite capacity `K`, the factory of `lotka_volterra` replaces
exponential growth of the prey by logistic growth. With `stochastic=True`, the factory of `sir` writes the epidemic as
random jumps of whole individuals, so the compiler infers a jump process instead of differential equations.

```python
from skcomplex.systems import create_simulation, describe

describe('sir').parameters
epidemic = create_simulation('sir', stochastic=True).prepare().run()
epidemic.problem_.form.value, epidemic.trajectory_.values('I.n').max()
```

#### Run settings of a ready-made system

The simulation keeps its run settings as parameters, so `set_params` changes them before you prepare it. Every
argument of `Simulation` is a run setting, such as `until`, `dt`, `solver`, `random_state`, and `initial_state`.

```python
from skcomplex.systems import create_simulation

simulation = create_simulation('lotka_volterra')
simulation.until, simulation.dt
simulation.set_params(dt=0.5, until=20.0, initial_state={'prey.n': 20.0})
trajectory = simulation.prepare().run().trajectory_
trajectory.times[-1], trajectory.values('prey.n')[0]
```

`run(until=...)` overrides the end time for one run and leaves `until` as it is.

```python continuation
simulation.run(until=5.0).trajectory_.times[-1], simulation.until
```

A new `initial_state` replaces the whole mapping that the catalog set, so repeat any tuned value you want to keep.

#### Simulating the system your own way

`simulation.system` is an ordinary `System`. Give it to a `Simulation` of your own to pick every setting yourself, such
as a fixed-step solver, a schedule of a parameter, or other observables.

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems import create_simulation

system = create_simulation('lotka_volterra').system
simulation = Simulation(system, solver=RK4Solver(dt=0.01), dt=0.5).prepare().run(until=30.0)
simulation.trajectory_.values('predators.n')[-1].round(3)
```

States and parameters are named by their dotted path, such as `prey.n` and `predation.a`. The labels of the elements
and interactions are in `system.elements` and `system.interactions`. The section [Simulating](#simulating) lists every
run setting.

### Composing systems

Build a larger system out of smaller ones, such as a landscape of a forest and a meadow, and couple their elements.
A system holds nested systems in `systems`, and an interaction reaches an element of a nested system by a
slash-separated path.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


class PredationInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        predator: Annotated[str, RoleSpec(states=('n',))],
        prey: Annotated[str, RoleSpec(states=('n',))],
        a: Annotated[float, ParameterSpec()] = 0.4,
        b: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label)
        self.predator = predator
        self.prey = prey
        self.a = a
        self.b = b

    def _get_laws(self, s):
        contacts = s.predator.n * s.prey.n
        return [RateLaw(s.prey.n, -s.a * contacts), RateLaw(s.predator.n, s.b * contacts)]


landscape = System(
    systems=[
        System('forest', elements=[PopulationElement('foxes', n=5)]),
        System('meadow', elements=[PopulationElement('rabbits', n=50)]),
    ],
    interactions=[PredationInteraction('hunt', predator='forest/foxes', prey='meadow/rabbits', a=0.02, b=0.01)],
)
trajectory = Simulation(landscape, dt=0.1).prepare().run(until=10).trajectory_
trajectory.values('meadow/rabbits.n')[-1].round(3)
```

The interaction `hunt` sits in the outer system and binds `forest/foxes` and `meadow/rabbits` to its roles. States and
parameters are named by their path, such as `meadow/rabbits.n` and `hunt.a`, in the trajectory, in initial values, and
in parameter changes.

A ready-made system from `create_system` nests in the same way. Give each nested system its own label with
`set_params`, since a system from the catalog has the label `system`.

```python
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

island = create_system('lotka_volterra').set_params(label='island')
town = create_system('sir').set_params(label='town')
trajectory = Simulation(System(systems=[island, town]), dt=0.1).prepare().run(until=5).trajectory_
trajectory.values('town/I.n')[-1].round(1)
```

## Build a custom system

Write a system of your own from elements, interactions, and laws. Start with the concepts and how to choose parts that
fit, then go to the section of the kind of system you need.

### Concepts

Learn the few words every custom system is written in. You describe what the system is, and `scikit-complexity` decides
how to solve it.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


growth = RateInteraction('growth', target='rabbits', state='n', rate=lambda s: 0.5 * s.target.n)
system = System(elements=[PopulationElement('rabbits', n=10)], interactions=[growth])
Simulation(system, dt=0.1).prepare().run(until=2).trajectory_.values('rabbits.n')[-1].round(2)

n = var('n')
RateLaw(n, 0.5 * n).prepare('growth').run({'n': 10.0}).value_
```

The element `rabbits` carries the state `n`, the interaction `growth` returns a `RateLaw` on it, and the system holds
both. The last line prepares one law on its own and runs it, which gives its rate at `n = 10`.

#### Parts

A system is made of **parts**: its elements, its interactions, the laws they return, and the system itself.

- An **element** is an entity that carries state, such as a particle, a population, a compartment, or a market.
- A **group** holds many identical elements of one type as arrays.
- An **interaction** couples elements through **roles** and describes its effect with **laws**.
- A **system** holds elements, interactions, nested systems, an optional space, and an optional time.

These concepts carry no field-specific meaning. A population, a particle, a firm, and a cell are all elements. So a
system in any field of science is a configuration of the same interface, and every ready-made system in
`skcomplex.systems` is a factory that builds such a configuration.

#### Two verbs

Every part speaks two verbs, and both return the part itself.

- `prepare(...)` validates the part and prepares it against what it works on.
- `run(...)` sets what the part contributes. An element, an interaction, or a system sets its laws in `laws_`. A law
  sets in `value_` its value at given values of the symbols it reads, such as the rate of a `RateLaw`.

A part must be prepared before it runs.

#### Laws

A **law** says how a state changes. Every law is a class, and these four are the most common.

- `RateLaw` adds a term to the time derivative of a state. Rates on one state add up.
- `NoiseLaw` drives a state with a Wiener process, independent unless laws name a shared or correlated one.
- `JumpLaw` fires a random event that changes states by fixed increments.
- `UpdateLaw` gives the next value of a state after one discrete step.

A **space** gives vector states, such as positions, their dimension. A **time** says whether the system evolves
continuously, in discrete steps, or not at all.

#### Runs

A run records **observables**, named values computed from the states at every frame, and checks the **invariants** it
is given.

`skcomplex.core.components` holds the parts. `skcomplex.core.engine` compiles a system, infers its mathematical form
from its laws, and builds fast numeric functions that its solvers advance. The laws decide the form, and the form
decides the default solver.

| Laws | Form | Default solver |
| --- | --- | --- |
| `RateLaw` | ordinary differential equations | `ScipyODESolver` with `LSODA` |
| `RateLaw` and `NoiseLaw` | stochastic differential equations | `EulerMaruyamaSolver` |
| `JumpLaw` | jump process | `GillespieSolver` |
| `UpdateLaw` | discrete map | `IterationSolver` |

### Choosing compatible components

Find the parts that fit together before you build a system, and read why a part does not fit. Choose them in this order:
time and space, then elements, then interactions, then the solver. Each choice narrows the next, and you can ask every
part what it needs before you build anything.

The [compatibility page](compatibility.md) lists every element and interaction of the core, with the kinds and states
each needs, so start there to find candidates. Then check a candidate in code, as this section shows.

#### Time and space

An interaction states the time and the space its laws run in, and an empty `space` means any space.
`get_requirements()` returns them.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction, MovementInteraction, RateInteraction
from skcomplex.core.components.spaces import ContinuousTime, DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.specs import StateSpec
from skcomplex.core.types import ElementKind

growth = RateInteraction('growth', target='rabbits', state='n', rate=lambda s: 0.5 * s.target.n)
needs = growth.get_requirements()
assert needs.time == (ContinuousTime,)
assert needs.space == ()
assert MapInteraction('step', target='rabbits', state='n', function=lambda s: s.target.n).get_requirements().time == (
    DiscreteTime,
)
[law.__name__ for law in needs.laws]
```

`RateInteraction` returns rates, so it needs continuous time and accepts any space. `MapInteraction` returns updates
and rules, so it needs discrete time.

#### Elements

An element carries the states you declare, and `get_states()` returns them with their specs.

```python continuation
class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


rabbits = PopulationElement('rabbits', n=40)
list(rabbits.get_states())
```

A `GroupElement` returns the states of its members. Every element is of one kind, and the kind is what a role accepts or
rejects.

- `Element` is of the kind `element`, one member with its states.
- `GroupElement` is of the kind `group`, many identical members. With `variable=True` it is a `variable-group`, whose
  members appear and disappear.
- A group on a space is a `site-group`, an `edge-group`, or a `hyperedge-group`, by what its members sit on. A
  `LatticeElement` and a `FieldElement` are site groups.

#### Interactions

An interaction names what each of its roles needs. `get_requirements().roles` holds, for every role, the kinds of
element it accepts and the states it reads for the arguments you passed. Compare them with what the element carries.
The kind `any` accepts every kind of element.

```python continuation
target = needs.roles['target']
target.states, target.elements
assert target.accepts(ElementKind.ELEMENT)
assert all(state in rabbits.get_states() for state in target.states)
```

The two agree, so the interaction fits the element. Some roles accept only some kinds. `MovementInteraction` moves the
agents of a group, so its `target` accepts a group and not a single element.

```python continuation
walk = MovementInteraction('walk', target='rabbits').get_requirements().roles['target']
walk.elements
assert walk.accepts(ElementKind.GROUP)
assert not walk.accepts(ElementKind.ELEMENT)
```

#### When a part does not fit

Read the validation report before you run. `PassCompiler(...).prepare(system).report_` holds every problem of the
system, each with a path, a message, and a fix. Here the rate reads the state `n` of a tank that carries `level`.

```python continuation
class TankElement(Element):
    def __init__(self, label, *, level: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.level = level


wrong = System(
    elements=[TankElement('tank')],
    interactions=[RateInteraction('growth', target='tank', state='n', rate=1.0)],
)
issue = PassCompiler().prepare(wrong).report_.issues[0]
issue.code, issue.path
issue.message
issue.fix
assert issue.code == 'role-element'
assert issue.message == (
    "The role 'target' of growth reads the state 'n', but the element tank bound to it carries level."
)
```

A role filled by an unsuitable element is the issue `role-element`. A role that rejects the kind of its element gets
the same issue, and a law in the wrong time gets `law-time`.

```python continuation
kinds = System(elements=[TankElement('tank')], interactions=[MovementInteraction('walk', target='tank')])
PassCompiler().prepare(kinds).report_.issues[0].message
times = System(
    elements=[TankElement('tank')],
    interactions=[RateInteraction('grow', target='tank', state='level', rate=1.0)],
    time=DiscreteTime(),
)
PassCompiler().prepare(times).report_.issues[0].message
```

If you do not read the report first, `prepare` of a `Simulation` raises it as an error that lists every problem.

```python continuation
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.errors import RoleElementError

try:
    Simulation(wrong, dt=0.1).prepare()
except RoleElementError as error:
    message = str(error)
message
```

An interaction of your own gets the same report.

- A `RoleSpec(states=('n',))` states the read in advance, so `get_requirements()` shows it.
- An interaction that declares nothing is still reported, when validation builds its laws and a law reads a state that
  the element does not have.
- A function you pass, such as the `rate` of `growth`, is covered in the same way.
- An error of another kind inside a law reaches you unchanged.

#### The solver

You can leave the solver out. The laws of a system decide its mathematical form, and the form decides the default
solver. To choose one yourself, pass it to the compiler and read the report before you run.

```python continuation
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.solvers import IterationSolver, RK4Solver

system = System(elements=[rabbits], interactions=[growth])
compile_system(system).form.value
assert PassCompiler(solver=RK4Solver()).prepare(system).report_.ok
assert not PassCompiler(solver=IterationSolver()).prepare(system).report_.ok
Simulation(system, dt=0.1).prepare().run(until=1).trajectory_.values('rabbits.n')[-1].round(1)
```

`RK4Solver` integrates the differential equations of `growth`, and `IterationSolver`, which iterates maps, does not.
The compatibility page lists what every solver supports.

### Defining elements

Write the element and interaction types of your own system. A type declares its states, parameters, and roles as
arguments of its `__init__`, and returns its laws from `_get_laws`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet
from sklearn.base import clone

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class SpeciesElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0):
        super().__init__(label)
        self.n = n


class BirthInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        rate: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _get_laws(self, s):
        return [RateLaw(s.target.n, s.rate * s.target.n)]


deer = SpeciesElement('deer', n=40.0)
birth = BirthInteraction('birth', target='deer', rate=0.2)
deer.n, clone(birth).rate
```

Every element type derives from `Element`, and every interaction type from `Interaction`. Both, like laws and systems,
derive from `BasePart`, which is a scikit-learn estimator. A type annotates each variable with `typing.Annotated` and a
spec from `skcomplex.core.specs`.

- `StateSpec` marks a state, `ParameterSpec` a parameter, and `RoleSpec` a role of an interaction. The default of the
  argument is the initial value of a state or the default of a parameter. A role has no default.
- The constructor takes the label first and every other argument by keyword. It stores each value unmodified under its
  own name.
- The domain of a state or a parameter is a Sage set, such as `RR`, `ZZ`, `CC`, `RealSet.closed_open(0, oo)`, or
  `FiniteEnumeratedSet(['S', 'I', 'R'])`.
- `_get_laws(s)` returns the laws of the type, and returns none by default.
- `get_params`, `set_params`, `clone`, and `repr` follow the scikit-learn estimator contract, so every part clones with
  its values.

Inside `_get_laws`, `s.rate` is the symbol of the parameter, and `s.target.n` is the symbol of the state of the element
bound to the role `target`. Parameters stay symbols, so a running simulation can change them without compiling again.

#### Constructor settings

A setting that is neither a state, a parameter, nor a role is an argument of `__init__` without a spec. Examples are a
switch between two sets of laws, a radius, or a layer.

```python continuation
from skcomplex.core.components.laws import JumpLaw


class SwitchedBirthInteraction(BirthInteraction):
    def __init__(self, label, *, target, rate=0.5, stochastic=False):
        super().__init__(label, target=target, rate=rate)
        self.stochastic = stochastic

    def _get_laws(self, s):
        if self.stochastic:
            return [JumpLaw(s.rate * s.target.n, {s.target.n: 1})]
        return super()._get_laws(s)


clone(SwitchedBirthInteraction('birth', target='deer', stochastic=True))
```

- The setting is a keyword-only argument with a default.
- `__init__` stores it unmodified under its own name, like every other argument.
- A subclass that adds a setting repeats the arguments of its base in its own `__init__`, so that `get_params` sees them
  all. An argument repeated without its annotation keeps the spec of the base.

#### States and parameters from data

An element whose variables depend on data, such as a fitted surrogate or a metabolic network, cannot name them in its
signature. `DeclaredElement` takes them in one argument instead.

```python continuation
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import DeclaredVariableSpec


class DecayElement(DeclaredElement):
    def _get_laws(self, s):
        return [RateLaw(getattr(s, name), -s.k * getattr(s, name)) for name in ('a', 'b')]


declared = {
    'a': DeclaredVariableSpec(1.0, StateSpec()),
    'b': DeclaredVariableSpec(2.0, StateSpec()),
    'k': DeclaredVariableSpec(0.5, ParameterSpec()),
}
decay = DecayElement('decay', variables=declared)
faster = clone(decay).set_params(k=1.0)
decay.get_value('k'), faster.get_value('k')
system = System(elements=[decay], time=ContinuousTime())
Simulation(system, dt=0.1).prepare().run(until=1.0).trajectory_.values('decay.b')[-1]
```

- `variables` maps every name to a `DeclaredVariableSpec`, which holds the initial value and a `StateSpec` or a
  `ParameterSpec`. The compiler treats a declared variable as it treats an annotated argument.
- A law reads a declared variable from its namespace, and `get_value` reads its value from the element.
- `set_params` takes the name of a declared variable and changes its value, and `clone` copies the variables.
- A group or a field takes the element class and gives `variables` in its `values`, next to the values per member.
- A subclass that derives its variables from other arguments overrides `_declare_variables`.

#### Writing laws

A law says how the states it targets change. Prepare a law on its own to see what it changes and reads, and run it to
get its value.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import BaseLaw, RateLaw

x, k = var('x k')
law = RateLaw(x, -k * x).prepare('decay')
issubclass(RateLaw, BaseLaw), law.targets_, sorted(map(str, law.reads_)), law.owner_
law.run({'x': 2.0, 'k': 0.5}).value_
```

Every law class derives from `BaseLaw`. A law takes the states it changes and the expressions it reads as arguments of
its `__init__`, and an optional `label` by keyword.

- `prepare(owner)` validates the arguments, such as a value that is a NumPy array instead of an expression. It sets
  `targets_`, the states the law changes, `reads_`, the symbols it reads, and `owner_`, the path of the element or
  interaction that returns it.
- `run(values)` sets `value_`, the value of the law at the values of the symbols it reads, given by name.
- Every law states what it requires of a system, such as continuous time for `RateLaw` and discrete time for
  `UpdateLaw`. Validation reports a system whose laws require different times.

A law of one state, such as `RateLaw`, takes the state as its first argument. A law that updates several states
together, such as `LocalRuleLaw`, takes a tuple of them. Laws that name their states in other values, such as the
increments of a `JumpLaw`, change the states they name.

A system is defined by its elements, its interactions, and the laws they return from `_get_laws(s)`, and by nothing
else.

- A value that only names an expression is a Python expression in the laws, such as `leverage = s.debt / s.equity`. A
  function of the namespace shares it between laws.
- What a run records or checks is an argument of the run. `Simulation(observables=[...])` records named values, and
  `Simulation(invariants={name: tolerance})` checks that the observables it names stay constant.

A law can also be a Python function of time and the current values, which covers rules that have no closed form. The
values are keyed by dotted state and parameter name.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import RoleSpec, StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


class CullInteraction(Interaction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.target = target

    def _get_laws(self, s):
        name = f'{self.target}.n'
        return [RateLaw(s.target.n, lambda t, values: -1.0 if values[name] > 50 else 0.0)]


culled = System(
    elements=[PopulationElement('deer', n=60.0)],
    interactions=[CullInteraction('cull', target='deer')],
)
round(float(Simulation(culled, dt=1.0).prepare().run(until=20).trajectory_.values('deer.n')[-1]))
```

#### Functions as laws

Write a law as a Python function when it has no symbolic form. Every law whose value the engine evaluates numerically
takes a function, and `CallSpec` sets how the function is called.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import CrossingLaw, JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import reduction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import CallSpec, ObservableSpec, ParameterSpec, StateSpec


class TankElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        level: Annotated[float, StateSpec()] = 1.0,
        rate: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label, visual=visual)
        self.level = level
        self.rate = rate

    def _get_laws(self, s):
        grow = CallSpec(lambda t, values: values['rate'] * values['level'], relative=True)
        full = CallSpec(lambda t, values: values['level'] - 2.0, relative=True)
        return [RateLaw(s.level, grow), CrossingLaw(full, {s.level: lambda t, values: 1.0}, direction=1)]


tanks = System(elements=[TankElement('a')], systems=[System('plant', elements=[TankElement('b')])])
trajectory = Simulation(tanks, dt=0.01).prepare().run(until=2).trajectory_
len(trajectory.events) > 0, trajectory.values('a.level')[-1] == trajectory.values('plant/b.level')[-1]
```

A function runs in Python, so on the Numba backend the laws of its kind fall back to Python with a `BackendWarning`.
This table lists the laws that take a function, and the form the function takes.

| Law | Argument | Function |
| --- | --- | --- |
| `RateLaw`, `NoiseLaw`, `UpdateLaw`, `BoundaryLaw` | `value` | `f(t, values)` or `CallSpec` |
| `JumpLaw` | `rate`, and each value of `changes` | `f(t, values)` or `CallSpec` |
| `JumpLaw` on a clock of waiting times | `waiting` | `f(rng, size=, t=, **values)` |
| `CrossingLaw` | `guard`, and each value of `reset` | `f(t, values)` or `CallSpec(f, relative=True)` |
| `FluxLaw` | `speed` | `f(t, values)`, which gives the wave speed of the Rusanov flux |
| `TransitionLaw` | `matrix` | `f(t, values)` or `CallSpec(f, relative=True)` |
| `reduction(function, expression)` | `function` | `f(array)` of the values of the members |
| `ObservableSpec` | `function` | `f(t, values)` |

`CallSpec(f, (s.x, s.k))` calls `f(t, x, k)` once per member. Three options change that.

- `relative=True` calls `f(t, values)` with the names of the component of the law, such as `x` for a state of the
  element and `target.x` for a state of the element bound to the role `target`. One element class then works under any
  label and inside any system, as the tanks `a` and `plant/b` above show.
- `random=True` passes the random stream of the simulation as the keyword `rng`, so one seed repeats the run.
- `vectorized=True` calls the function once per evaluation with one array per argument, instead of once per member.

`arguments=None` calls `f(t, values)` with the dotted names, like a plain function, and takes the options `random` and
`vectorized`.

A random function draws from `rng`, so two runs with the same seed agree.

```python continuation
class WalkerElement(TankElement):
    def _get_laws(self, s):
        return [UpdateLaw(s.level, CallSpec(lambda t, level, rng: level + rng.normal(), (s.level,), random=True))]


def walk(seed):
    system = System(elements=[WalkerElement('a')], time=DiscreteTime())
    return Simulation(system, random_state=seed).prepare().run(until=5).trajectory_.values('a.level')


np.array_equal(walk(1), walk(1))
```

A vectorized function takes the arrays of a whole group at once, and `reduction` applies a function of an array to the
values of the members.

```python continuation
class HerdElement(TankElement):
    def _get_laws(self, s):
        return [RateLaw(s.level, CallSpec(lambda t, level, rate: rate * level, (s.level, s.rate), vectorized=True))]


class MedianElement(TankElement):
    def _get_laws(self, s):
        return [RateLaw(s.level, reduction(np.median, s.level) - s.level)]


shapes = []
for element in (HerdElement, MedianElement):
    herd = System(elements=[GroupElement('herd', element=element, size=3, values={'level': [1.0, 2.0, 4.0]})])
    shapes.append(Simulation(herd, dt=0.01).prepare().run(until=1).trajectory_.values('herd.level').shape)
shapes
```

The change of a jump can be a function too. Here every jump doubles the level.

```python continuation
class DoublerElement(TankElement):
    def _get_laws(self, s):
        return [JumpLaw(s.rate, {s.level: CallSpec(lambda t, values: values['level'], relative=True)})]


doubling = Simulation(System(elements=[DoublerElement('a')]), random_state=0).prepare().run(until=10)
float(np.log2(doubling.trajectory_.values('a.level')[-1])).is_integer()
```

An observable can be a function of the time and the values too. The run records it at every frame.

```python continuation
def total_level(t, values):
    return values['a.level'] + values['a.rate']


observables = [ObservableSpec('sum', function=total_level)]
observed = Simulation(System(elements=[TankElement('a')]), dt=0.5, observables=observables)
observed.prepare().run(until=1).trajectory_.values('sum').shape
```

Some laws need symbols by nature, and each says so when given a function.

- `EquationLaw` is solved symbolically, so a function goes in a `RateLaw` of the highest derivative.
- `GeneratorLaw` and `DissipatorLaw` build quantum operators from symbols.
- A `FluxLaw` takes the derivative of its flux, for the wave speeds and for the critical points of the Godunov flux, so
  a function goes in `speed=` or in a `RateLaw`.
- A crossing, a matrix, and a jump change read no delays.
- A reset of a renewal stays symbolic, because it shares its draws with the waiting time.

#### Symbolic expressions

Write laws as Sage expressions, so they can be inspected, differentiated, and solved.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo, sqrt
from sage.sets.real_set import RealSet

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import where
from skcomplex.core.specs import ParameterSpec, RoleSpec


class CrowdingInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        capacity: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 100.0,
    ):
        super().__init__(label)
        self.target = target
        self.capacity = capacity

    def _get_laws(self, s):
        n = s.target.n
        return [RateLaw(n, where(n > s.capacity, -sqrt(n - s.capacity), 0.1 * n * exp(-n / s.capacity)))]


CrowdingInteraction('crowding', target='deer')
```

The symbols in `s` are Sage symbols that carry the domains of their variables, vector states are Sage vectors, and
mathematical functions come from Sage. Sage has no condition-based piecewise expression or symbolic remainder. So
`skcomplex.core.components.laws.utils` adds `where`, `either`, `both`, and `mod`, which every backend lowers, LAMMPS
included.

Systems written with SymPy move to Sage as follows. A SymPy expression in a law raises a `ValidationError`, and
`expression._sage_()` converts one that is already built.

| SymPy | Sage and `skcomplex` |
| --- | --- |
| `sympy.sqrt`, `sympy.exp`, `sympy.sin`, and other functions | `sqrt`, `exp`, `sin`, and others from `sage.all__sagemath_symbolics` |
| `sympy.Symbol('x', positive=True)` | `var('x', domain='positive')` from `sage.all__sagemath_symbolics` |
| `sympy.Matrix([...])` and `a.dot(b)` | `vector([...])` from `sage.all__sagemath_symbolics` and `a.dot_product(b)` |
| `sympy.Piecewise((a, condition), (b, True))` | `where(condition, a, b)` |
| `condition1 \| condition2` and `condition1 & condition2` | `either(condition1, condition2)` and `both(condition1, condition2)` |
| `sympy.Mod(x, m)` | `mod(x, m)` |
| `sympy.diff(f, x)` and `f.subs(x, y)` | `f.diff(x)` and `f.subs({x: y})` |
| `sympy.Eq(a, b)`, `.lhs`, and `.rhs` | `a == b`, `.lhs()`, and `.rhs()` |

#### State domains

Give a state labels or angles instead of plain numbers. Besides real, nonnegative, and integer values, a state takes the
labels of a finite alphabet, a `FiniteEnumeratedSet`, or angles, a state with `StateSpec(period=2 * pi)`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import pi
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.specs import StateSpec

HEALTH = FiniteEnumeratedSet(['susceptible', 'infected', 'recovered'])


class PersonElement(Element):
    def __init__(self, label, *, health: Annotated[str, StateSpec(domain=HEALTH)] = 'susceptible'):
        super().__init__(label)
        self.health = health


class RotorElement(Element):
    def __init__(self, label, *, theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0):
        super().__init__(label)
        self.theta = theta


PersonElement('ada').health, RotorElement('rotor').theta
```

Values are stored as floats. Labels are stored as their index, and angles are wrapped into one turn after every frame.
Trajectories return labels, and `Simulation(..., checked=True)` raises an error naming any state that leaves its
domain.

#### Complex states

Give a state complex values, such as the amplitude of an oscillation, with the domain `CC`.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import CC, I

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class PhaseElement(Element):
    def __init__(
        self,
        label,
        *,
        z: Annotated[complex, StateSpec(domain=CC)] = 1.0,
        omega: Annotated[float, ParameterSpec()] = 2.0,
    ):
        super().__init__(label)
        self.z = z
        self.omega = omega

    def _get_laws(self, s):
        return [RateLaw(s.z, I * s.omega * s.z)]


z = Simulation(System(elements=[PhaseElement('osc')]), dt=0.5).prepare().run(until=10).trajectory_.values('osc.z')
z.dtype == np.complex128, bool(np.allclose(np.abs(z), 1.0, atol=1e-4))
```

A state or parameter of the domain `CC` takes complex values. Its laws use the imaginary unit `I`, arithmetic, the
elementary functions over the complex plane, and `real_part`, `imag_part`, `conjugate`, `abs`, and `arg`.

- The compiler writes every complex law as the laws of its real and imaginary parts, which every solver, stepper, and
  backend runs as a real system.
- A real state takes only real values, so `abs(z)` or `real_part(z)` turns a complex value into one.
- A condition compares the modulus or a part, since complex values have no order.
- Trajectories and frames return complex arrays. Their `.real`, `.imag`, `np.abs`, and `np.angle` are the real views
  that the analyses take.
- Agent states stay real, and the LAMMPS solver raises an error naming a complex state.

#### Equations of any order

Write an equation of motion as it reads, with second or higher derivatives. Newton's second law for a pendulum,
`m l θ'' = -m g sin θ`, is an `EquationLaw` on a state of order 2.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import pi, sin

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.specs import ParameterSpec, StateSpec


class PendulumElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi, order=2)] = 0.5,
        theta_d1: float = 0.0,
        m: Annotated[float, ParameterSpec()] = 1.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
        l: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.theta_d1 = theta_d1
        self.m = m
        self.g = g
        self.l = l

    def _get_laws(self, s):
        return [EquationLaw(s.m * s.l * derivative(s.theta, 2) == -s.m * s.g * sin(s.theta))]


bob = PendulumElement('bob', theta_d1=0.5)
list(bob.get_params()), bob.theta_d1
```

Laws give the highest time derivative of a state, and a state declares how high that is with `StateSpec(order=k)`.
The angular velocity is the companion state `theta_d1`, and the compiler reduces the equation to first order for the
solvers.

- A companion is a constructor argument without a spec, named `<state>_d<j>`. Its value is the initial value of the
  derivative of order `j`, so `PendulumElement('bob', theta_d1=0.5)` starts the pendulum with an angular velocity.
- `StateSpec(initial=...)` gives the initial values of the derivatives instead, for a type without companion arguments.
- On a state of order `k`, rate and noise laws contribute to the `k`-th derivative and add up. So separate interactions
  supply separate forces, voltages, or growth terms. `RateLaw(s.target.x, -s.k * s.target.x / s.target.m)` from a
  spring and `RateLaw(s.target.x, -s.c * derivative(s.target.x, 1) / s.target.m)` from a damper superpose.
- `derivative(x, j)` reads the companion `x_d<j>` below the state's order. An `EquationLaw` may contain the derivative
  of the state's own order, linearly. The compiler solves all equations of a system jointly for those derivatives,
  which couples, for example, the two angles of a double pendulum.
- Any order works the same way, such as Sprott's chaotic jerk equation `x''' = -A x'' - x' + |x| - 1` on a state of
  order 3.
- Difference equations keep past values with `StateSpec(lags=m)`. `lag(x, j)` reads the value `j` steps before the
  current one, and an update law gives the next value. So `UpdateLaw(s.x, s.x + lag(s.x, 1))` with one lag is the
  Fibonacci recurrence.

`SymbolicAnalysis('equations')` returns these equations in the order they were written.
`SymbolicAnalysis('solve', use_initial_state=True)` takes the initial values of every derivative and returns a solution
only when it has a closed form.

### Kinds of dynamics

Find the laws, elements, spaces, and solvers of the kind of dynamics your system needs, and see how a compiler turns
the laws into a problem a solver can run.

```python
from skcomplex.core.engine.compile import BaseCompiler, PassCompiler
from skcomplex.systems import create_system

compiler = PassCompiler().prepare(create_system('sir')).run()
isinstance(compiler, BaseCompiler), compiler.report_.ok, compiler.problem_.form.value
```

Some kinds of dynamics, such as lattice dynamics or operator dynamics, are needed by few systems. None of them is a
package of its own. Its laws, elements, and spaces are parts of the core, and the passes that compile it, its solvers,
and its visuals are parts of the engine, beside those of every other kind.

#### Where each kind lives

| Kind | Core | Engine |
| --- | --- | --- |
| agents | `RuleLaw`, `MatchLaw` | the agents passes, `MesaSolver` |
| lattices | `LocalRuleLaw`, `LocalRuleInteraction`, `DriveLaw`, `CascadeLaw`, `ExtremalLaw`, `LatticeElement`, `LatticeSpace`, `ObjectiveSpec`, `SamplerSpec` | `LatticeSolver`, `MonteCarloSolver` |
| finitely generated groups | `CayleySpace` | `CayleyVisual` |
| rewriting | `ProductionLaw`, `SubstitutionLaw`, `WordElement`, `PatchElement` | `RewritingCompiler`, `RewritingSolver`, `GeometryVisual` |
| operator | `GeneratorLaw`, `DissipatorLaw`, `HilbertSpec` | a system compiler, `UnitarySolver`, `LindbladSolver`, `JumpTrajectoriesSolver` |
| fields | `FieldElement`, `FluxLaw`, `MeshSpace`, `laplacian` and the other operators | the fields passes, `IMEXEulerSolver`, `CrankNicolsonSolver`, `SSPRKSolver` |
| kinetics | `EncounterLaw`, `moment` | the encounter passes, `KineticSolver` |
| fractional | `StateSpec(order=...)` in `(0, 1)`, `FractionalNoiseLaw` | the fractional passes, `FractionalABMSolver`, `GrunwaldLetnikovSolver`, `FractionalEulerMaruyamaSolver` |
| topology | `TopologyRequestSpec` | the topology passes |
| particles | `PointElement`, `GradientInteraction`, `PairCouplingInteraction` | `VelocityVerletSolver`, `Yoshida4Solver`, `LammpsSolver` |
| control | `ParameterSpec(adaptive=True, bounds=...)` | `PontryaginSolver`, `CollocationSolver` |

Every compiler derives from `BaseCompiler` and speaks the two verbs of every part.

- `prepare(system)` validates its settings and the system, and sets `report_`, the issues it found.
- `run(initial_state)` raises the errors of the report, sets `problem_`, the problem, and sets `solver_`, the solver it
  names for the problem.

`Simulation` and `compile_system` take the compiler as the `compiler` setting. The default `PassCompiler` compiles every
system but operator ones through its passes, which lower each law to the problem of its form. For a system that needs
another compiler, it names that compiler.

- An operator system takes `compiler=OperatorCompiler()`.
- A rewriting system takes `compiler=RewritingCompiler()`.
- A package of your own can bring the compiler of a form of its own, with the solver it names.

#### Writing a law

A new law derives from `BaseLaw`. It stores its arguments under their own names, and it implements three methods.

- `_get_targets` returns the states it changes, which `prepare` sets as `targets_`.
- `_get_requirements` returns a `RequirementSpec` of the kinds of time and space it runs in.
- `_run` sets `value_`, its value at given values of the symbols it reads, which `prepare` sets as `reads_`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.specs import ParameterSpec, RequirementSpec, StateSpec


class DecayLaw(BaseLaw):
    def __init__(self, target, rate, *, label=''):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _check(self):
        pass

    def _get_targets(self):
        return (self.target,)

    @classmethod
    def _get_requirements(cls):
        return RequirementSpec(time=(ContinuousTime,))

    def _run(self, inputs):
        self.value_ = self._evaluate(-self.rate * self.target, inputs)


n, k = var('n k')
law = DecayLaw(n, k).prepare('atom')
law.targets_, sorted(map(str, law.reads_)), law.owner_, law.run({'n': 2.0, 'k': 0.5}).value_
```

The law checks, prepares, and runs on its own. But the passes and the system compilers are fixed lists of the engine,
so no form takes a law of a new class in a system. The compiler reports it with the code `laws-mixed`, and a new kind of
dynamics is a change to the engine.

```python continuation
class NucleusElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0, k: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.n = n
        self.k = k

    def _get_laws(self, s):
        return [DecayLaw(s.n, s.k)]


report = PassCompiler().prepare(System(elements=[NucleusElement('atom')])).report_
[issue.code for issue in report.errors]
```

### General interactions

Most processes you want to describe already exist as a general interaction. You pick the class that does the
mathematical operation you need, bind it to your elements, and pass the functions that make it your process.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

decay = RateInteraction('decay', target='stock', state='n', rate=lambda s: -0.5 * s.target.n)
system = System(elements=[CompartmentElement('stock', n=8.0)], interactions=[decay])
round(float(Simulation(system, dt=0.01).prepare().run(until=2.0).trajectory_.values('stock.n')[-1]), 3)
```

`skcomplex.core.components.interactions` holds one general class per mathematical operation: a rate, a gradient, a
map, a coupling, a jump, a transition, a choice, a movement, a transfer, a grouping. Find the operation you need in the
list, then follow the link to its subsection.

- Change a state continuously by a rate, towards a goal, or with noise: `RateInteraction` and
  `RelaxationInteraction`, in [Rates and relaxation](#rates-and-relaxation).
- Move states down a potential, or write the mechanics of a Lagrangian or a Hamiltonian: `GradientInteraction`,
  `EulerLagrangeInteraction`, and `HamiltonianInteraction`, in [Gradients and mechanics](#gradients-and-mechanics).
- Conserve, carry, or spread a quantity over a field: `ConservationInteraction`, `AdvectionInteraction`, and
  `DiffusionInteraction`, in [Conservation laws](#conservation-laws).
- Update a state in steps, search by trial and error, or run a cellular automaton or a cascade: `MapInteraction`,
  `SignAdjustmentInteraction`, `LocalRuleInteraction`, and `CascadeInteraction`, in [Discrete time](#discrete-time).
- Couple members over the edges of a lattice or a network, over a group, or over close pairs: `CouplingInteraction`
  and `PairCouplingInteraction`, in [Coupling over a structure](#coupling-over-a-structure).
- Fire random events or chemical reactions: `JumpInteraction` and `ReactionInteraction`, in
  [Jumps and reactions](#jumps-and-reactions).
- Move a member between discrete values, such as susceptible and infected: `TransitionInteraction`, in
  [Transitions between values](#transitions-between-values).
- Give the encounters of a kinetic system: `EncounterInteraction`, in
  [Encounters of a kinetic system](#encounters-of-a-kinetic-system).
- Reset states when a condition crosses zero, or at chosen times: `CrossingInteraction` and `ImpulseInteraction`, in
  [Crossings and impulses](#crossings-and-impulses).
- Sample configurations by Monte Carlo: `SamplingInteraction`, in [Monte Carlo sampling](#monte-carlo-sampling) and
  [Row scores and zero temperature](#row-scores-and-zero-temperature).
- Set a state from the time or the random stream of the step: `DriveLaw`, in
  [Drives that read the step](#drives-that-read-the-step).
- Run an operation as symbolic laws or as agent rules: the `form` and `schedule` arguments, in
  [Agent and symbolic forms](#agent-and-symbolic-forms).
- Choose, imitate, learn, or evolve: `ChoiceInteraction`, `ImitationInteraction`, `ReinforcementInteraction`, and
  `EvolutionInteraction`, in [Choice and learning](#choice-and-learning).
- Move agents, transfer amounts, create and remove agents, form groups, match partners, or rewire a network:
  `MovementInteraction`, `TransferInteraction`, `LifeCycleInteraction`, `GroupingInteraction`,
  `MatchingInteraction`, `FeedbackInteraction`, and `TopologyInteraction`, in
  [Agents, groups, and exchange](#agents-groups-and-exchange).
- Place agents in three dimensions or in regions such as census tracts: `ContinuousSpace` and `MovementInteraction`,
  in [Agents in three dimensions and on region layers](#agents-in-three-dimensions-and-on-region-layers).
- Write laws with Hill functions, delays, pulses, or group reductions: `skcomplex.core.components.laws.utils`, in
  [Law utilities](#law-utilities).
- Evolve a tensor state of a field: `StateSpec(tensor=...)`, in [Tensor fields](#tensor-fields).
- Do something no general class covers: the binding bases, in [When no general class fits](#when-no-general-class-fits).

#### How the general classes are configured

A class names what it does, never what a state means. No core class or argument names a force, a position, a price,
an infection, or a harvest. You configure a class by the roles it binds, the names of the states it reads and writes,
and the functions you pass.

- `target` is the role whose states the laws change. `sources` are roles the laws only read.
- `state`, `states`, `read`, and `write` name the states, so one class serves any meaning. A force is a rate on a
  state of order 2, and a mass is its `divisor`.
- Functions are callables of the namespace `s` that return symbolic expressions, such as
  `rate=lambda s: -s.k * s.target.n`. Where an argument allows it, a number stands for a constant function.
- `divisor` divides what a rate, a gradient, or a pair coupling adds to the highest derivative, such as a mass, a
  friction, or a capacitance. It is a named argument, not part of the function, so a backend such as LAMMPS can read
  it.
- A variant is a string checked against the constants of its module, such as `mode='increment'`. A variant that
  carries an algorithm of its own is a frozen spec instead, such as the rule of a matching.

#### What enters the core

A class enters the core when it is a fundamental paradigm, such as a variational principle, a conservation law, a
Markov jump process, a reaction network, or imitation dynamics, and when it applies in principle to other domains than
the one that first needs it. How many fields use it today does not decide it.

The case of one field is a thin subclass of a general class in the catalog, even when it only renames arguments. It
lives with the one system that uses it, or with the code its field shares when several systems use it. These
subclasses double as examples of how to build a part of your own, and the page of each system shows them. The
interactions of `lotka_volterra`, for example, are all rate interactions:

```python
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.systems import create_system

system = create_system('lotka_volterra')
{type(part).__name__: isinstance(part, RateInteraction) for part in system.interactions}
```

#### Elements of an interaction

The general interactions act on a few general elements, so you rarely need an element class of your own.

```python
from skcomplex.core.components.elements import CompartmentElement, PointElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

system = System(
    elements=[CompartmentElement('stock', n=5.0), PointElement('point', x_d1=[1.0, 0.0])],
    space=ContinuousSpace(n_dim=2),
)
trajectory = Simulation(system, dt=1.0).prepare().run(until=2.0).trajectory_
trajectory.values('stock.n')[-1], trajectory.values('point.x')[-1].tolist()
```

- `Element` declares states and parameters. `GroupElement` holds many identical members as arrays. `FieldElement` and
  `LatticeElement` place one member at every site of a space.
- `CompartmentElement` is a stock with one state `n`, such as a species, a compartment, or a queue. Since states are
  declared per class, a system renames `n` by a thin subclass.
- `DensityElement` is the density `f` of one activity class of a kinetic system.
- `PointElement` is a member with a vector state `x` of order 2 in a continuous space. Its first derivative is `x_d1`,
  so a backend such as LAMMPS reads the velocity of a point from its declared states.

#### Continuous time

These classes write differential equations, with or without noise.

##### Rates and relaxation

Use `RateInteraction` to add any rate to a state, as the decay at the top of this section does, and
`RelaxationInteraction` to pull a state towards a goal.

`RateInteraction(target, state, rate, sources=(), noise=None, divisor=None, hurst=None)` adds `rate(s)`, and
`noise(s) dW`, to the highest derivative of a state of any order. A Hurst index `hurst` makes the noise fractional.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RelaxationInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

smoothing = RelaxationInteraction('smoothing', target='belief', state='n', goal=10.0, time=0.5)
system = System(elements=[CompartmentElement('belief', n=0.0)], interactions=[smoothing])
round(float(Simulation(system, dt=0.01).prepare().run(until=5.0).trajectory_.values('belief.n')[-1]), 3)
```

`RelaxationInteraction(target, state, goal, time, noise=None)` relaxes a state towards a goal at the rate
`(goal - x) / time`. With noise, it is an Ornstein-Uhlenbeck process.

##### Gradients and mechanics

Use `GradientInteraction` when a state moves down a potential. On a state of order 1 it is a gradient flow:

```python
from skcomplex.core.components.elements import CompartmentElement, PointElement
from skcomplex.core.components.interactions import GradientInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

well = GradientInteraction('well', target='walker', states='n', potential=lambda s: s.target.n**2 / 2)
flow = System(elements=[CompartmentElement('walker', n=1.0)], interactions=[well])
round(float(Simulation(flow, dt=0.01).prepare().run(until=1.0).trajectory_.values('walker.n')[-1]), 4)
```

On a state of order 2, such as the position of a point, the same class gives Newton's equations:

```python continuation
spring = GradientInteraction('spring', target='point', states='x', potential=lambda s: s.target.x * s.target.x / 2)
newton = System(elements=[PointElement('point', x=[1.0])], interactions=[spring], space=ContinuousSpace(n_dim=1))
round(float(Simulation(newton, dt=0.01).prepare().run(until=3.1416).trajectory_.values('point.x')[-1][0]), 3)
```

`GradientInteraction(target, states, potential, sources=(), divisor=None, variational=False, conserved=False)` also
gives the variational or conserved dynamics of a field. The divisor is a name, a function of the member, or a mapping
of one divisor per state.

Use `EulerLagrangeInteraction` when you know the Lagrangian of a mechanical system, such as a pendulum:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import cos

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import EulerLagrangeInteraction
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class AngleElement(Element):
    def __init__(self, label, *, q: Annotated[float, StateSpec(order=2)] = 0.1, q_d1: float = 0.0):
        super().__init__(label)
        self.q = q
        self.q_d1 = q_d1


lagrangian = lambda s: derivative(s.target.q, 1) ** 2 / 2 + cos(s.target.q)
swing = EulerLagrangeInteraction('swing', target='bob', states='q', lagrangian=lagrangian)
system = System(elements=[AngleElement('bob')], interactions=[swing])
round(float(Simulation(system, dt=0.01).prepare().run(until=3.2).trajectory_.values('bob.q')[-1]), 2)
```

`EulerLagrangeInteraction(target, states, lagrangian, dissipation=None, forcing=None)` writes the equations of motion
of a Lagrangian, with a Rayleigh dissipation and a generalized forcing. The equations of one coordinate from several
interactions add, so a coupling is an interaction whose Lagrangian holds only the coupling energy.

Use `HamiltonianInteraction` when you know the Hamiltonian, such as a planet around a sun:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import HamiltonianInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import Yoshida4Solver
from skcomplex.core.specs import StateSpec


class PlanetElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 0.0,
        px: Annotated[float, StateSpec()] = 0.0,
        py: Annotated[float, StateSpec()] = 1.0,
    ):
        super().__init__(label)
        self.x = x
        self.y = y
        self.px = px
        self.py = py


def kepler(s):
    return (s.target.px**2 + s.target.py**2) / 2 - 1 / sqrt(s.target.x**2 + s.target.y**2)


coordinates, momenta = ('x', 'y'), ('px', 'py')
gravity = HamiltonianInteraction('gravity', target='sun', coordinates=coordinates, momenta=momenta, hamiltonian=kepler)
system = System(elements=[PlanetElement('sun')], interactions=[gravity])
trajectory = Simulation(system, solver=Yoshida4Solver(dt=0.01), dt=1.0).prepare().run(until=6.2832).trajectory_
round(float(trajectory.values('sun.x')[-1]), 6)
```

`HamiltonianInteraction(target, coordinates='q', momenta='p', hamiltonian=None)` writes Hamilton's equations
`q' = dH/dp` and `p' = -dH/dq` as rate laws of canonical coordinates and momenta. These are states of order 1, and
they may be angles. The rates of one state add, so the Hamiltonians of separate interactions sum.

`Yoshida4Solver` integrates a separable Hamiltonian at fourth order by three velocity Verlet steps. It first splits the
states into two blocks whose rates read only the other block. It keeps the energy bounded over long runs.

`InvariantAnalysis` computes Poisson brackets over the canonical pairs. It decides whether expressions given as
`ObservableSpec`s are conserved, and checks a `SymmetrySpec` against the Hamiltonian or the Lagrangian, with its
Noether charge:

```python continuation
from skcomplex.core.analysis import InvariantAnalysis
from skcomplex.core.specs import ObservableSpec, SymmetrySpec

momentum = ObservableSpec('angular_momentum', lambda s: s.sun.x * s.sun.py - s.sun.y * s.sun.px)
InvariantAnalysis(expressions=[momentum]).prepare(system).run().conserved_
rotation = SymmetrySpec('rotation', lambda s: {'sun.x': -s.sun.y, 'sun.y': s.sun.x})
InvariantAnalysis('symmetry', symmetry=rotation).prepare(system).run().charge_
```

##### Conservation laws

Use these classes for a quantity of a field that is only moved around, never created or lost, except by a source. The
example runs Burgers' flux, a drift, and diffusion on a periodic finite-volume field, and the total stays the same in
each case.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import AdvectionInteraction, ConservationInteraction, DiffusionInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u


space = LatticeSpace((50,), spacing=0.1, boundary='periodic')
values = {'u': 1.0 + np.sin(2 * np.pi * np.arange(50) / 50)}
interactions = {
    'burgers': ConservationInteraction('burgers', target='wave', state='u', flux=lambda s: s.target.u**2 / 2),
    'drift': AdvectionInteraction('drift', target='wave', state='u', drift=lambda s: 1.0),
    'heat': DiffusionInteraction('heat', target='wave', state='u', coefficient=0.1),
}
totals = {}
for label, interaction in interactions.items():
    field = FieldElement('wave', element=CellElement, space=space, values=values, discretization='finite-volume')
    system = System(elements=[field], interactions=[interaction])
    total = Simulation(system, dt=0.01).prepare().run(until=0.5).trajectory_.values('wave.u')[-1].sum()
    totals[label] = round(float(total), 6)
totals
```

- `ConservationInteraction(target, state, flux, source=None, limiter='minmod', riemann='rusanov')` conserves a state of
  a finite-volume field under a flux and a source. The flux laws of one field form one system. With `riemann=None`, it
  lowers to the divergence of the flux on spectral and finite-difference fields.
- `AdvectionInteraction(target, state, drift)` carries a state by a drift `a`, which is the flux `a u`.
- `DiffusionInteraction(target, state, coefficient)` is the flux minus a coefficient times the gradient.

#### Discrete time

These classes update states in steps, as maps, searches, cellular automata, and cascades. Use `MapInteraction` to set
the next value of a state from the current one, such as the logistic map:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import MapInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class ValueElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.x = x


logistic = MapInteraction('map', target='value', state='x', function=lambda s: 3 * s.target.x * (1 - s.target.x))
system = System(elements=[ValueElement('value')], interactions=[logistic], time=DiscreteTime())
Simulation(system).prepare().run(until=2).trajectory_.values('value.x').tolist()
```

`MapInteraction(target, state, function, sources=(), mode='replace', schedule='synchronous', form='symbolic')` sets
`x(t+1) = f(s)`, or `x + f(s)` with `mode='increment'`. The schedule decides the order of the members:

- `'synchronous'` updates all members at once.
- `'sequential'` updates them one after the other.
- `'random-sequential'` updates them in random order.
- `'member'` updates them in the activation of every member, in the agent form.

Use `SignAdjustmentInteraction` when a state searches for a target by trial and error, such as a price that rises
while demand exceeds supply:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import SignAdjustmentInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class QuoteElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.x = x


signal = lambda s: 2 - s.target.x
search = SignAdjustmentInteraction('search', target='quote', state='x', signal=signal, step=0.5, upper=2.0)
system = System(elements=[QuoteElement('quote')], interactions=[search], time=DiscreteTime())
Simulation(system).prepare().run(until=3).trajectory_.values('quote.x').tolist()
```

`SignAdjustmentInteraction(target, state, signal, step=0.1, mode='multiplicative', lower=None, upper=None)` sets
`x (1 + step sign(signal))`, or `x + step sign(signal)` with `mode='additive'`, kept within `[lower, upper]`.

Use `LocalRuleInteraction` for a cellular automaton and `CascadeInteraction` for a cascade. The helper `run` below
places a row of five cells and runs one interaction on it:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import CascadeInteraction, LocalRuleInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace, VonNeumannNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec, TotalisticSpec


class CellElement(Element):
    def __init__(self, label, *, c: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.c = c


def run(interaction, values, until):
    cells = LatticeElement('cells', element=CellElement, space=LatticeSpace(5), values={'c': values})
    system = System(elements=[cells], interactions=[interaction], time=DiscreteTime())
    return Simulation(system, dt=until, random_state=0).prepare().run(until=until)


parity = LocalRuleInteraction(
    'parity', target='cells', state='c', rule=TotalisticSpec({1: 1}), neighborhood=VonNeumannNeighborhood(n_dim=1)
)
run(parity, np.eye(5)[2], 1).state_.tolist()
```

`LocalRuleInteraction(target, state, rule, neighborhood)` runs a cellular automaton from a rule table, a totalistic
rule, or a function of the ordered neighbourhood.

```python continuation
around = VonNeumannNeighborhood(n_dim=1, center=True)
toppling = CascadeInteraction('toppling', target='cells', state='c', threshold=2, share=1, neighborhood=around)
simulation = run(toppling, [0.0, 1.0, 3.0, 1.0, 0.0], 1)
simulation.state_.tolist(), [event.size for event in simulation.trajectory_.events]
```

`CascadeInteraction(target, state, threshold, share, mode, drive=None, neighborhood=None)` first drives the members.
It then relaxes every member at or above a threshold by a reset, a redistribution, or a random share, and records each
cascade as an event.

#### Coupling over a structure

Use these classes when members influence each other: neighbours on a lattice or a network, every member of a group, or
pairs of points closer than a cutoff. Kuramoto oscillators, for example, couple every member of a group through the
mean of a sine:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class OscillatorElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec()] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        return [RateLaw(s.theta, s.omega)]


rng = np.random.default_rng(0)
values = {'theta': rng.uniform(-np.pi, np.pi, 50), 'omega': rng.normal(0.0, 0.5, 50)}
oscillators = GroupElement('oscillators', element=OscillatorElement, size=50, values=values)
kuramoto = CouplingInteraction(
    'kuramoto',
    target='oscillators',
    read='theta',
    write='theta',
    function=lambda s: 2.0 * sin(s.other.theta - s.member.theta),
    reduce='mean',
    mode='group',
)
system = System(elements=[oscillators], interactions=[kuramoto])
phases = Simulation(system, dt=1.0).prepare().run(until=10).trajectory_.values('oscillators.theta')[-1]
round(float(np.abs(np.exp(1j * phases).mean())), 2)
```

`CouplingInteraction(target, source=None, read, write, function, reduce='sum', mode='edges', ...)` gives member `i` a
reduction over `j` of `function(s)`. Inside the function, `s.member` reads `i`, `s.other` reads `j`, and `s.edge`
reads the edge between them.

- It works over the edges of a lattice or a network, over the whole group or a separate `source` group
  (`mode='group'`), or over the next member of an ordered group.
- The reductions are sum, mean, max, min, normalized mean, and circular mean.
- The weights are an edge state, a matrix, or a function.
- It adds to a rate (`form='rate'`), or sets the next value with `combine(s, reduction)` (`form='map'`).

Use `PairCouplingInteraction` for forces between pairs of points, such as a Morse potential between two atoms:

```python
import numpy as np
from sage.all__sagemath_symbolics import exp

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.interactions import PairCouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

morse = PairCouplingInteraction('morse', 'pair', 'pair', cutoff=3.0, potential=lambda s, r: 2 * (1 - exp(1 - r)) ** 2)
pair = GroupElement('pair', element=PointElement, size=2, values={'x': [[0.0, 0.0], [1.5, 0.0]]})
system = System(elements=[pair], interactions=[morse], space=ContinuousSpace(n_dim=2))
positions = Simulation(system, dt=0.1).prepare().run(until=5).trajectory_.values('pair.x')
distances = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
round(float(distances.min()), 2), round(float(distances.max()), 2)
```

`PairCouplingInteraction(first, second, cutoff, read, write, function=None, potential=None, divisor=None,
period=None)` adds to both members of every pair within the cutoff either `function(s)`, or minus the gradient of
`potential(s, r)` at their distance `r`. With several states in `write`, the function returns one value per state.

#### Events and discrete states

These classes change states by discrete, often random, events.

##### Jumps and reactions

Use `JumpInteraction` for random events that change states by fixed amounts, such as arrivals to a queue:

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import JumpInteraction, ReactionInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

arrivals = JumpInteraction('arrivals', target='queue', changes={'n': 1}, rate=lambda s: 2.0)
queue = System(elements=[CompartmentElement('queue')], interactions=[arrivals])
counts = Simulation(queue, dt=1.0, random_state=0).prepare().run(until=5).trajectory_.values('queue.n')
all(count == round(count) for count in counts)
```

`JumpInteraction(target, changes, rate=None, probability=None, waiting=None, reset=None)` fires stochastic events that
change states by signed amounts. They fire at a rate, with a probability per step, or after drawn waiting times.

Use `ReactionInteraction` for mass action between species, such as an epidemic written as two reactions. The total
population stays the same:

```python continuation
infection = ReactionInteraction('infection', reactants={('S', 'n'): 1, ('I', 'n'): 1}, products={('I', 'n'): 2}, k=5e-4)
recovery = ReactionInteraction('recovery', reactants={('I', 'n'): 1}, products={('R', 'n'): 1}, k=0.1)
sir = System(
    elements=[CompartmentElement('S', n=990), CompartmentElement('I', n=10), CompartmentElement('R')],
    interactions=[infection, recovery],
)
round(float(Simulation(sir, dt=1.0).prepare().run(until=50).trajectory_.states[-1].sum()), 6)
```

`ReactionInteraction(target=None, reactants, products, k, propensity=None, modifiers=None, mode='rate')` runs as
rates, as jumps (`mode='jump'`), or with Langevin noise. A species is a pair of a path and a state, or a state name of
the `target` alone. A reaction with a group or field target therefore reacts in every member.

##### Transitions between values

Use `TransitionInteraction` when a member moves between discrete values, such as states of health or of ownership:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import TransitionInteraction
from skcomplex.core.components.laws.utils import neighbors, where
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class HolderElement(Element):
    def __init__(self, label, *, state: Annotated[int, StateSpec(domain=NN)] = 1):
        super().__init__(label)
        self.state = state


move = TransitionInteraction(
    'move', target='holders', state='state', source=1, destination=2, probability=lambda s: 0.2
)
holders = GroupElement('holders', element=HolderElement, size=1000)
system = System(elements=[holders], interactions=[move], time=DiscreteTime())
values = Simulation(system, random_state=0).prepare().run(until=3).trajectory_.values('holders.state')
round(float((values[-1] == 1).mean()), 1)
```

`TransitionInteraction(target, state, source, destination, rate=None, probability=None, condition=None, matrix=None,
...)` moves a discrete state from a value, a set of values, or the values a function accepts, to another value. It
fires at a rate, with a probability per step, when a condition holds, or by a transition or an emission matrix. With
`after`, it fires after a residence time.

A rate can read the neighbours of a member, so spreading along edges with recovery takes two transitions:

```python continuation
contact = TransitionInteraction(
    'contact',
    target='sites',
    state='state',
    source=0,
    destination=1,
    rate=lambda s: 4.0 * neighbors(where(s.target.other.state == 1, 1, 0), reduce='mean'),
)
recovery = TransitionInteraction('recovery', target='sites', state='state', source=1, destination=0, rate=lambda s: 1.0)
sites = LatticeElement('sites', element=HolderElement, space=LatticeSpace((20, 20)))
system = System(elements=[sites], interactions=[contact, recovery])
values = Simulation(system, dt=1.0, random_state=0).prepare().run(until=10).trajectory_.values('sites.state')
values[-1].mean() > 0.3
```

The systems of the field `biology/epidemiology` in the catalog build on such a subclass.

##### Encounters of a kinetic system

Use `EncounterInteraction` for the bilinear encounters of a kinetic system, whose elements are densities on activity
classes:

```python
import numpy as np

from skcomplex.core.components.elements import DensityElement, GroupElement
from skcomplex.core.components.interactions import EncounterInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

kernel = np.zeros((3, 1, 3))
kernel[:, :, 0] = 1.0
gather = EncounterInteraction(
    'gather', candidate='candidates', field='field', rate=lambda s: 1.0, kernel=lambda s: kernel
)
system = System(
    elements=[
        GroupElement('candidates', element=DensityElement, size=3, on='sites', values={'f': [0.2, 0.3, 0.5]}),
        GroupElement('field', element=DensityElement, size=1, values={'f': 1.0}),
    ],
    interactions=[gather],
    space=LatticeSpace(3, boundary='fixed'),
)
densities = Simulation(system, dt=0.5).prepare().run(until=4.0).trajectory_.values('candidates.f')
np.round(densities[-1], 2).tolist()
```

`EncounterInteraction(candidate, field, rate, kernel, outcome)` gives encounters that are conservative, destructive,
or proliferative.

##### Crossings and impulses

Use `CrossingInteraction` to act when a condition is met, and `ImpulseInteraction` to act at chosen times. The example
wraps a clock back to zero each time it reaches 1, and kicks it forward once at time 1.5:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import CrossingInteraction, ImpulseInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class ClockElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, 1.0)]


wrap = CrossingInteraction(
    'wrap', target='clock', guard=lambda s: s.target.x - 1, reset={'x': lambda s: 0}, direction=1
)
kick = ImpulseInteraction('kick', target='clock', reset={'x': lambda s: s.target.x + 0.25}, times=(1.5,))
system = System(elements=[ClockElement('clock')], interactions=[wrap, kick])
events = Simulation(system, dt=0.5).prepare().run(until=2.2).trajectory_.events
[(event.label, round(event.time, 3)) for event in events]
```

- `CrossingInteraction(target, guard, reset=None, direction=0, terminal=False)` fires an event when a guard crosses
  zero. The event resets states, or ends the run.
- `ImpulseInteraction(target, reset, times=(), period=None, count=None)` resets states at given or periodic times.

##### Monte Carlo sampling

Use `SamplingInteraction` to sample the configurations of a lattice, such as the spins of an Ising magnet at low
temperature, which align:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class SpinElement(Element):
    def __init__(self, label, *, s: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.s = s


objective = lambda s: -s.target.s * neighbors(s.target.other.s)
ising = SamplingInteraction('ising', target='spins', state='s', objective=objective, intensity=1.0)
spins = LatticeElement('spins', element=SpinElement, space=LatticeSpace((8, 8)))
system = System(elements=[spins], interactions=[ising], time=DiscreteTime())
abs(Simulation(system, random_state=0).prepare().run(until=20).trajectory_.states[-1].mean()) > 0.9
```

`SamplingInteraction(target, state, objective, move='flip', intensity=1.0)` runs Monte Carlo sweeps. A move is accepted
from the change of an objective with probability `exp(-intensity * change)`. With `move='cluster'`, it makes Wolff
cluster flips instead. To sample an energy at a temperature, pass the energy as the objective and one over the
temperature as the intensity.

#### Row scores and zero temperature

Use a row score when the objective is a function of a whole row of a lattice, rather than a sum of local terms. The
example flips one bit of each row at a time at zero temperature, so the score of every row never grows:

```python
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import RowScoreSpec, StateSpec


class BitElement(Element):
    def __init__(self, label, s: Annotated[int, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 1):
        super().__init__(label)
        self.s = s


def flip_one(row, random):
    site = random.integers(row.size)
    row[site] = 1 - row[site]
    return row


walk = SamplingInteraction(
    'walk', target='bits', state='s', objective=RowScoreSpec(np.sum), move=flip_one, intensity=float('inf')
)
bits = LatticeElement('bits', element=BitElement, space=LatticeSpace((3, 20)))
system = System(elements=[bits], interactions=[walk], time=DiscreteTime())
scores = Simulation(system, random_state=0).prepare().run(until=50).trajectory_.values('walk.score')
bool(np.all(np.diff(scores, axis=0) <= 0))
```

`SamplingInteraction` takes a `RowScoreSpec(function, proposals=1)` as its objective. The function reads the values of
one row of a state on a `LatticeSpace`, its last axis, and the parameters that the function names.

- Every unit of time proposes `proposals` moves per row through the move. Each is accepted from the change of the
  score of the row, and the rows take separate draws.
- A named move changes one random site of a row. A function move receives the row.
- A trajectory records the score of every row as `<label>.score`.

The intensity may be infinite, which is the zero temperature of the thin subclass in physics. The Metropolis rule then
accepts a move only when the objective does not grow, for local objectives and row scores alike.

#### Drives that read the step

Use a `DriveLaw` when a state is set from outside, by the time of the step or by a random draw, such as replaying an
ensemble of recorded configurations:

```python
import numpy as np
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import DriveLaw

ensemble = np.zeros((38, 20, 3))
replay = DriveLaw(var('x'), lambda values, *, time: ensemble[min(int(time), 37)])
replay.prepare().targets_
```

`DriveLaw` passes the time of the step to a function that names `time`, and the random generator to a function that
names `random`. A function can then set a lattice state to one member of an ensemble per step, as the system
`fuzzy_oil_drop` does, and hold the last member after the end.

#### Agent and symbolic forms

A class whose operation agent systems also use takes `form='symbolic'` or `form='agent'`. Choose the agent form when
your members are agents that need to read the system's history, storage, or their own neighbours and cell.

```python
from types import SimpleNamespace

from skcomplex.core.components.interactions import MapInteraction

symbolic = MapInteraction('map', target='walkers', state='x', function=lambda s: s.target.x + 1)
agent = MapInteraction('map', target='walkers', state='x', function=lambda agent, system: agent.x + 1, form='agent')
type(agent.prepare().run(SimpleNamespace()).laws_[0]).__name__, symbolic.form
```

- The symbolic form lowers to symbolic laws, which every solver compiles.
- The agent form lowers to an agent `RuleLaw` in its `phase`. Its functions take the agent and the system view,
  `f(agent, system)`, so they read the system's history and storage and the agent's neighbours and cell.
- The form is an argument, not a guess, because the engine probes laws with empty namespaces and cannot tell an agent
  group from a symbolic one.

**The member schedule.** Use `schedule='member'` when every agent should take all of its actions in one go, such as
moving and then counting its steps:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MapInteraction, MovementInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class WalkerElement(Element):
    def __init__(self, label, *, steps: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.steps = steps


walk = MovementInteraction('walk', target='walkers', step='neighbor', schedule='member')
count = MapInteraction(
    'count', target='walkers', state='steps', function=lambda a, m: a.steps + 1, schedule='member', form='agent'
)
walkers = GroupElement('walkers', element=WalkerElement, size=3)
system = System(elements=[walkers], interactions=[walk, count], space=LatticeSpace((5, 5)))
Simulation(system, random_state=0).prepare().run(until=4).trajectory_.agents('walkers', 4).values['steps'].tolist()
```

An agent rule may name the role whose members run it, `RuleLaw(function, members='target')`. The Mesa solver runs it
in the activation of every member of that role, after the rules of the member's element and the member rules of the
interactions listed before it. `MapInteraction`, `TransitionInteraction`, `MovementInteraction`,
`TransferInteraction`, and `LifeCycleInteraction` take `schedule='member'` for it. The movement, transfers,
transitions, and life cycle of one agent then run together as one step of that agent, in the order of the system's
interactions.

#### Choice and learning

These classes let members choose between options, copy each other, learn from payoffs, and evolve strategies. Use
`ChoiceInteraction` when every member picks an option by its utility, such as members that prefer to agree with the
mean:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import ChoiceInteraction
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class ChooserElement(Element):
    def __init__(self, label, *, option: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.option = option


utility = lambda s, option: 2.0 * option * mean(s.target.option)
choice = ChoiceInteraction('choice', target='members', state='option', options=(-1, 1), utility=utility)
members = GroupElement('members', element=ChooserElement, size=500)
system = System(elements=[members], interactions=[choice], time=DiscreteTime())
values = Simulation(system, random_state=0).prepare().run(until=10).trajectory_.values('members.option')
round(float(np.mean(values[-1])), 1)
```

`ChoiceInteraction(target, state, options, utility, rule='logit', intensity=1.0)` draws an option for every member by
the logit, Metropolis, Fermi, or best rule from `utility(s, option)`, or `utility(agent, system, option)` in the agent
form.

Use `ImitationInteraction` when members copy their neighbours, such as voters that follow the local majority:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import ImitationInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class VoterElement(Element):
    def __init__(self, label, *, state: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.state = state


initial = np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
majority = ImitationInteraction('majority', target='voters', rule='majority', form='probability')
voters = LatticeElement('voters', element=VoterElement, space=LatticeSpace(12), values={'state': initial})
system = System(elements=[voters], interactions=[majority], time=DiscreteTime())
Simulation(system, random_state=0).prepare().run(until=1).trajectory_.values('voters.state')[-1].tolist()
```

`ImitationInteraction(target, state, payoff, options, rate, rule='random', form='rate')` copies the state of a
neighbour at random, by majority, from the best, from a random better neighbour, or by the Fermi rule. It runs as
jumps, as updates with a probability per step (`form='probability'`), or as an agent rule.

Use `ReinforcementInteraction` when members keep a score per option and learn from payoffs:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import ReinforcementInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class LearnerElement(Element):
    def __init__(
        self,
        label,
        *,
        first: Annotated[float, StateSpec()] = 0.0,
        second: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.first = first
        self.second = second


learning = ReinforcementInteraction(
    'learning', target='learners', scores=('first', 'second'), payoff=lambda s, k: float(k + 1), rate=0.5
)
learners = GroupElement('learners', element=LearnerElement, size=2)
system = System(elements=[learners], interactions=[learning], time=DiscreteTime())
trajectory = Simulation(system, random_state=0).prepare().run(until=10).trajectory_
first, second = trajectory.values('learners.first')[-1][0], trajectory.values('learners.second')[-1][0]
round(float(first), 2), round(float(second), 2)
```

`ReinforcementInteraction(target, scores, payoff, rate=0.1, rule='average')` moves the score of every option towards
its realized payoff, or adds the payoff with `rule='cumulative'`.

Use `EvolutionInteraction` for a genetic algorithm or evolutionary dynamics over strategies. In the example, the
fitness of a bit string is its number of ones, and the mean rises over the generations:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import EvolutionInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, StateSpec


class BitStringElement(Element):
    def __init__(
        self,
        label,
        *,
        strategy: Annotated[object, StateSpec()] = 0.0,
        fitness: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.strategy = strategy
        self.fitness = fitness

    def _get_laws(self, s):
        return [RuleLaw(self.score, phase='scoring')]

    def score(self, agent, system):
        agent.fitness = float(np.sum(agent.strategy))


initial = np.random.default_rng(0).integers(0, 2, size=(40, 20)).astype(float)
agents = GroupElement('agents', element=BitStringElement, size=40, values={'strategy': initial})
evolution = EvolutionInteraction('evolution', target='agents', phase='evolution')
schedule = PhasesScheduleSpec([PhaseSpec('scoring'), PhaseSpec('evolution')])
system = System(elements=[agents], interactions=[evolution], time=DiscreteTime(schedule=schedule))
trajectory = Simulation(system, random_state=0).prepare().run(until=30).trajectory_
final = np.array(trajectory.agents('agents', 30).values['strategy'].tolist())
final.sum(axis=1).mean() > initial.sum(axis=1).mean() + 4
```

`EvolutionInteraction(target, strategy, fitness, selection='tournament', replacement='generational', crossover,
mutation, ...)` replaces strategies by selection, crossover, and mutation every `period` steps.

- The selection is by tournament, roulette, proportional, or rank.
- The replacement replaces the whole generation, or replaces by Moran, death-birth, truncation, or the best mutant.

#### Agents, groups, and exchange

These classes move agents, pass amounts between them, create and remove them, gather them into groups, pair them, and
change the network they live on.

##### Movement

Use `MovementInteraction` to move agents. Here two walkers each take one step to a neighbouring cell:

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

walkers = GroupElement('walkers', element=Element, size=2, location=[(2, 2), (2, 2)])
walk = MovementInteraction('walk', target='walkers', step='neighbor')
system = System(elements=[walkers], interactions=[walk], space=LatticeSpace((5, 5), neighborhood='von_neumann'))
trajectory = Simulation(system, random_state=0).prepare().run(until=1).trajectory_
[int(abs(cell - 2).sum()) for cell in trajectory.agents('walkers', 1).locations]
```

`MovementInteraction(target, step='neighbor', ...)` moves agents by a neighbour, Gaussian, Lévy, best-cell, or user
step, with a condition, an acceptance, exclusion, and a cost.

##### Transfers

Use `TransferInteraction` to move an amount from one holder to another, so that the total is kept:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class TankElement(Element):
    def __init__(self, label, *, level: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.level = level


flow = TransferInteraction('flow', source='upper', target='lower', state='level', amount=lambda s: s.source.level / 2)
system = System(elements=[TankElement('upper', level=4.0), TankElement('lower')], interactions=[flow])
trajectory = Simulation(system).prepare().run(until=1.0).trajectory_
round(float(trajectory.values('upper.level')[-1] + trajectory.values('lower.level')[-1]), 6)
```

`TransferInteraction(source, target, state, into=None, amount, ...)` moves amounts of states between members, the
agents of one cell, a `Source()`, and a `Sink()`. In the symbolic form these are transfer laws, which the trajectory
records as named flows. In the agent form they move between paired agents.

Transfers within one group can follow the group order, as in the exclusion process of particles hopping along a lane.
Every site holds at most one particle, and the number of particles stays the same:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.laws.utils import bernoulli
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class SiteElement(Element):
    def __init__(self, label, *, occupied: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.occupied = occupied


hops = TransferInteraction(
    'hops',
    source='lane',
    target='lane',
    state='occupied',
    amount=lambda s: s.source.occupied * (1 - s.target.occupied) * bernoulli('hop', 0.5),
    offset=1,
)
lane = GroupElement('lane', element=SiteElement, size=20, values={'occupied': np.tile([1.0, 0.0], 10)})
system = System(elements=[lane], interactions=[hops], time=DiscreteTime())
trajectory = Simulation(system, dt=1, random_state=0).prepare().run(until=10).trajectory_
set(trajectory.values('lane.occupied').sum(axis=1))
```

- With `offset=k`, a symbolic transfer within one group moves amounts along the edges of the group order, from every
  member to the member `k` places along it, which `namespace.target` reads.
- The `boundary` is `'periodic'` for a ring, or `'fixed'` for a line, whose last member moves nothing.
- With a `rate`, a `reverse_rate` adds jumps that move the amounts back.
- A draw in an amount is one draw per edge, read by both of its ends. The parallel update of the exclusion process
  therefore keeps one particle per site.
- A leg end `'<role>.<state>'` names a state of any role. One transfer, such as the circuit of money of a stock-flow
  consistent economy, can then move amounts between several holders, and a subclass declares the parameters its
  amounts read.

##### Birth and death

Use `LifeCycleInteraction` to create and remove agents. Here a cell of size 8 divides twice, and each division splits
the size between parent and offspring:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import LifeCycleInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, *, size: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.size = size


cells = GroupElement('cells', element=CellElement, size=1, values={'size': 8.0}, variable=True)
division = LifeCycleInteraction('division', target='cells', birth=1.0, split=('size',))
system = System(elements=[cells], interactions=[division], space=LatticeSpace((3, 3)))
Simulation(system, random_state=0).prepare().run(until=2).trajectory_.agents('cells', 2).values['size'].tolist()
```

`LifeCycleInteraction(target, birth, death, rate=None, offspring=1, placement='here', inherit=(), split=(), ...)`
creates offspring by a condition, a probability, or a rate. Offspring values are inherited, given, split, or mutated.
It also removes or renews agents.

##### Grouping and matching

Use `GroupingInteraction` when agents meet in small groups and act together, such as the majority vote of Galam's
majority rule:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import GroupingInteraction
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class VoterElement(Element):
    def __init__(self, label, *, s: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.s = s


def adopt_majority(members, system):
    majority = float(2 * sum(member.s for member in members) > len(members))
    for member in members:
        member.s = majority


voters = GroupElement('voters', element=VoterElement, size=3, values={'s': [1.0, 1.0, 0.0]})
galam = GroupingInteraction('galam', target='voters', groups='partition', size=3, act=adopt_majority)
trajectory = Simulation(System(elements=[voters], interactions=[galam])).prepare().run(until=1).trajectory_
trajectory.agents('voters', 1).values['s'].tolist()
```

`GroupingInteraction(target, groups='pairs', act, size=2)` draws pairs, a random partition, neighbourhoods, or groups
by a state every step. It then calls `act(members, system)` on every group.

Use `MatchingInteraction` to pair requesters with providers, such as buyers and sellers in a market, and
`FeedbackInteraction` to set parameters from what happened in a round:

```python
from skcomplex.core.components.interactions import FeedbackInteraction, MatchingInteraction
from skcomplex.core.specs import QueueSpec, StableMatchingSpec

queue = MatchingInteraction('queue', requesters='buyers', providers='sellers', rule=QueueSpec())
stable = MatchingInteraction('stable', requesters='buyers', providers='sellers', rule=StableMatchingSpec())
controller = FeedbackInteraction('controller', rule=lambda system: None, phase='policy')
queue.reports, type(stable.rule).__name__, controller.phase
```

`MatchingInteraction(requesters, providers, rule, limit, quantity, budget, transfers, on_match)` pairs requesters
with providers by the algorithm its rule names, then moves the states of `transfers`. The rules are frozen specs of
`skcomplex.core.specs` that hold only the settings of their algorithm:

- `RandomSearchSpec(tries, clearing)`, the default.
- `PartnerSpec(candidates, choice, fallback, capacity, score, clearing)`.
- `PartnerSwitchSpec(sample, intensity, partner)`.
- `QueueSpec(order)`.
- `IterativeMatchingSpec(adjustment, iterations, tolerance)`.
- `ContinuousMatchingSpec(arrival)`.
- `BatchMatchingSpec()`.
- `StableMatchingSpec(requester_preference, provider_preference, clearing)`.

The matching runs each algorithm as its own method, chosen by the type of its rule, so a new algorithm is a new spec
and a method. A `FeedbackInteraction(rule)` reads observables once per round of its phase and sets parameters. The
section [Phases, markets, institutions, and learning](#phases-markets-institutions-and-learning) runs both.

##### Network topology

Use `TopologyInteraction` when the network itself changes. Here every node of a ring of five loses an edge in one
step:

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=20)
links = GroupElement('links', element=Element, on='edges', variable=True, capacity=20)
expiry = TopologyInteraction('expiry', target='nodes', action='remove-edge')
system = System(elements=[nodes, links], interactions=[expiry], space=NetworkSpace(nx.cycle_graph(5)))
trajectory = Simulation(system, random_state=0).prepare().run(until=1).trajectory_
len(trajectory.network(0).sources), len(trajectory.network(1).sources)
```

`TopologyInteraction(target=None, action='add-edge', rate=None, probability=None, toward=None, ...)` adds or removes
nodes and edges of a network, and rewires edges, preferentially towards a weight. Without a target, it adds one node
per step from outside the network.

##### When no general class fits

The binding bases stay the classes to subclass for an operation the general classes do not cover:

- `Interaction` for laws of its own over any roles.
- `NeighborInteraction` over the edges of a lattice or a network.
- `PairInteraction` over the pairs within a cutoff.
- `HyperedgeInteraction` over the members of hyperedges.

The sections [Groups of many elements](#groups-of-many-elements), [Networks](#networks), and
[Pair interactions](#pair-interactions) show them.

#### Agents in three dimensions and on region layers

Agents can live in a box of three dimensions, or move between named regions such as census tracts.

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

divers = GroupElement('divers', element=Element, size=4)
swim = MovementInteraction('swim', target='divers', step='gaussian', scale=0.1)
space = ContinuousSpace(extent=(10.0, 5.0, 2.0), boundary='periodic')
system = System(elements=[divers], interactions=[swim], space=space)
trajectory = Simulation(system, random_state=0).prepare().run(until=5).trajectory_
trajectory.agents('divers', 5).locations.shape
```

`MesaSolver` runs agents in a `ContinuousSpace` of two or three sides. Placement, `MovementInteraction` steps, radius
queries, the minimum image of a periodic box, and the locations of a trajectory use every side, and
`system.dimensions` returns every side.

- The solver rejects a continuous space with any other number of sides, or without an extent, before the first frame.
- The compile report holds the issue `location-dimension` for an initial location whose length differs from the
  dimension of its space.
- `AgentsVisual` draws agents in three dimensions as `Points3D`.

To place agents in regions, give the space layers of polygons:

```python
from shapely import Point, box

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

census = [box(0.0, 0.0, 4.0, 4.0), box(5.0, 0.0, 9.0, 4.0)]
places = [box(1.0, 6.0, 3.0, 8.0)]
people = GroupElement('people', element=Element, size=3, location=[census[0], census[0], census[1]])
outing = MovementInteraction('outing', target='people', step=lambda agent, system: system.region('places', 0))
space = ContinuousSpace(extent=(10.0, 10.0), regions={'census': census, 'places': places})
system = System(elements=[people], interactions=[outing], space=space)
trajectory = Simulation(system, random_state=0).prepare().run(until=1).trajectory_
all(places[0].contains(Point(point)) for point in trajectory.agents('people', 1).locations)
```

`ContinuousSpace(regions=...)` holds layers of regions in a space of two sides. It is a mapping from the name of a
layer to a sequence of Shapely polygons in the coordinates of the space. `prepare` keeps one `STRtree` per layer in
`regions_`. For a region that is not a valid polygon, it raises `SpaceError`, which names the layer and the region.

- A polygon is a location in `GroupElement(location=...)`, `SystemView.create`, and `AgentView.move_to`. The solver
  draws a uniform point inside it from the generator of the simulation. It chooses a triangle of the constrained
  Delaunay triangulation of the polygon with a probability proportional to its area.
- `AgentView.regions_here(layer)` returns the indices of the regions of a layer that contain the agent.
- `SystemView.region(layer, index)` returns the polygon of a region, and `SystemView.regions(layer)` the number of
  regions of a layer.
- A `MovementInteraction` step that returns the polygon of a region moves an agent into it.
- `AgentsVisual` draws every layer once as static `LineStrips2D` under the agents.

**Real maps.** The polygons of a space are in the coordinates of the space, and the library handles no coordinate
reference system. Census shapefiles usually come in longitude and latitude, where every distance, such as a radius, is
in degrees. Reproject them to a metric system first. With GeoPandas,
`geopandas.read_file(path).to_crs(epsg=32604).geometry` gives the polygons of a shapefile in metres for a Hawaiian
island. Then shift them so that no coordinate is negative, since a box runs from 0 to its sides. The next example does
the same with a plain local projection of one census polygon.

```python
import math

import numpy as np
import shapely
from shapely import Polygon
from shapely.affinity import translate

EARTH_RADIUS = 6371000.0
tract = Polygon([(-159.6, 21.9), (-159.4, 21.9), (-159.4, 22.1), (-159.6, 22.1)])


def project(points):
    scale = math.cos(math.radians(22.0))
    return EARTH_RADIUS * np.radians(points) * np.array([scale, 1.0])


projected = shapely.transform(tract, project)
low_x, low_y, _, _ = projected.bounds
tract_in_metres = translate(projected, -low_x, -low_y)
[round(value, -3) for value in tract_in_metres.bounds]
```

The system `kauai_population` places one agent per resident of the census regions of the synthetic island of its
package, and `boids_3d` flies in a periodic cube.

```python
from skcomplex.systems import create_system

system = create_system('kauai_population', populations=(10, 5, 0))
system.elements[0].size, list(system.space.regions)
create_system('boids_3d', n_boids=10).space.extent
```

#### Law utilities

`skcomplex.core.components.laws.utils` gives the functions laws are commonly written with, and every backend lowers
them.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws.utils import circular_mean, clip, hill, logistic, pulse_train, sign

x, t, theta = var('x t theta')
float(hill(x, 2, 3).subs(x=2)), float(logistic(x, 1, 2).subs(x=1))
[clip(x, 0, 1).subs(x=value) for value in (-2, 0.5, 3)], [sign(x).subs(x=value) for value in (-2, 0, 3)]
[float(pulse_train(t, 10, 1, count=2).subs({t: time})) for time in (10.5, 20.5)], circular_mean(theta)
```

- `where`, `either`, `both`, and `mod` select, combine conditions, and take remainders.
- `hill(x, half, n)`, `logistic(x, center, width)`, `clip(x, low, high)`, and `sign(x)` are pointwise functions.
- `minimum` and `maximum` of several expressions compare them. Of one expression, they reduce it over a group.
- `total`, `mean`, `count`, and `circular_mean` reduce over a group. With `by=`, they reduce over the members that
  share the value of a state, such as the members of one household or one cell.
- `delay1`, `delay3`, and `smooth` are the first- and third-order delays and the smoothing of stock-and-flow systems.
- `pulse_train(t, period, width, height, start, count=None)` is a train of rectangular pulses, finite with a count.

#### Tensor fields

Use a tensor state when a field carries a matrix at every site, such as the orientation of a liquid crystal or a
stress. The example stirs a symmetric traceless tensor by the flow it drives, and diffusion and decay shrink it:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import (
    advection,
    inverse_laplacian,
    laplacian,
    leray_projection,
    tensor_divergence,
)
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class StirredElement(Element):
    def __init__(self, label, Q: Annotated[object, StateSpec(tensor='traceless')] = 0.0):
        super().__init__(label)
        self.Q = Q

    def _get_laws(self, s):
        force = tensor_divergence(s.Q)
        velocity = leray_projection(force.apply_map(lambda item: inverse_laplacian(-item)))
        return [RateLaw(s.Q, -s.Q.apply_map(lambda item: advection(item, velocity)) + s.Q.apply_map(laplacian) - s.Q)]


size = 16
spacing = 2 * np.pi / size
x, y = np.meshgrid(np.arange(size) * spacing, np.arange(size) * spacing, indexing='ij')
start = np.stack([0.1 * np.cos(x).ravel(), 0.1 * np.sin(y).ravel()], axis=1)
film = FieldElement(
    'film',
    element=StirredElement,
    space=LatticeSpace((size, size), spacing=spacing),
    values={'Q': start},
    discretization='spectral',
)
trajectory = Simulation(System(elements=[film]), dt=0.01).prepare().run(until=0.2).trajectory_
bool(np.max(np.abs(trajectory.values('film.Q')[-1])) < np.max(np.abs(start)))
```

`StateSpec(tensor=symmetry)` declares a rank-2 tensor state of a field on a lattice of 2 or 3 dimensions. The symmetry
is `'full'`, `'symmetric'`, `'antisymmetric'`, or `'traceless'`, which means symmetric and traceless.

- The element stores only the independent components, such as `Q_xx` and `Q_xy` for a symmetric traceless state in 2
  dimensions. Initial values give one value per independent component.
- Laws read `s.Q` as a Sage matrix whose dependent entries are written in terms of the independent ones. A `RateLaw`
  on it takes a matrix.

The operations on vectors and tensors are:

- `vector_gradient(v)` is the matrix `partial(v_i, j)`, and `tensor_divergence(T)` the vector of the sums of
  `partial(T_ij, j)` over `j`.
- `symmetric_part(M)` and `antisymmetric_part(M)` split a matrix, so the strain rate of a velocity is
  `symmetric_part(vector_gradient(u))`.
- `T.apply_map(laplacian)` is the Laplacian of a matrix, entry by entry, as `v.apply_map(laplacian)` is that of a
  vector.
- The transport of a tensor or a vector by a flow advects it, rotates it with the vorticity, and aligns it with the
  strain rate. It is an expression of `advection`, `vector_gradient`, `symmetric_part`, and `antisymmetric_part`, and a
  law subtracts it from its rate.
- `leray_projection(v)` is the divergence-free part of a vector, `v - gradient(inverse_laplacian(divergence(v)))`. It
  is exact on spectral fields of periodic lattices. A velocity odd about the middle of an axis, set by the `symmetry`
  of the field, gives walls there.

The `boundary` of a field takes a matrix of the symmetry of the state as the fixed value `('fixed', matrix)`, which
gives every edge of the lattice that matrix. `PassCompiler` reports a rate law that breaks the declared symmetry with
the code `tensor-symmetry`, and a tensor operation given the wrong shape with the code `law-domain`.

### Connected components

Assemble a system from components that connect through ports, as engineers do with circuits and plants. Each
component states equations and leaves open which variable each equation determines. So one resistor gives a current
under a voltage source and a voltage under a current source.

```python
from typing import Annotated, ClassVar

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec

CHEMICAL = ConnectorSpec('chemical', potential='mu', flow='n')


class ReservoirElement(Element):
    ports: ClassVar = {'a': CHEMICAL}

    def __init__(
        self,
        label,
        *,
        a_mu: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        a_n: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        level: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.a_mu = a_mu
        self.a_n = a_n
        self.level = level

    def _get_laws(self, s):
        return [EquationLaw(s.a_n, s.a_mu - s.level)]


system = System(
    elements=[ReservoirElement('first', level=1.0), ReservoirElement('second', level=3.0)],
    interactions=[ConnectionInteraction('node', members=['first', 'second'], ports=['a', 'a'])],
)
trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
assert trajectory.values('first.a_mu')[-1] == 2.0
assert trajectory.values('second.a_n')[-1] == -1.0
trajectory.values('first.a_mu')[-1], trajectory.values('second.a_n')[-1]
```

The two reservoirs share one node. The node makes their potentials equal and their flows sum to zero, so the potential
settles between the two levels and a flow runs from one reservoir into the other.

#### Algebraic states

`StateSpec(algebraic=True)` declares a state that no law integrates and that the equations of the system determine at
every time.

- An algebraic state has order 1 and no lags.
- Its initial value is the starting guess of a nonlinear solve.
- A law other than an `EquationLaw` that targets it raises `AlgebraicStateError`.
- The trajectory records it at every frame, so observables and visuals read it like any other state.
- The simulation keeps it at its initial value in `state_`, since the solvers integrate only the other states.
- A law given as a function that reads values by name sees that initial value. Read it through a symbolic law instead.

#### Ports and connections

`ConnectorSpec(name, potential, flow)` names the variables of a port, and the core holds no named connector. An element
type declares its ports in the class variable `ports`, a mapping from the name of a port to its connector. It takes one
algebraic state per variable of a port, named `<port>_<variable>`. A flow is positive into the element.

`ConnectionInteraction(label, members=..., ports=...)` is one node.

- It binds the elements of the node through a variadic role, and names the port of each in the same order.
- It returns the equations that make the potentials equal and the flows sum to zero.
- A node with many ports is one connection.
- A port that no connection names has a zero flow.

A circuit takes one connector and one element class per component. Here a capacitor discharges through a resistor.

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import ConnectionInteraction
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ConnectorSpec, ParameterSpec, StateSpec

ELECTRICAL = ConnectorSpec('electrical', potential='v', flow='i')


class TwoPinElement(Element):
    ports: ClassVar = {'p': ELECTRICAL, 'n': ELECTRICAL}

    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
    ):
        super().__init__(label)
        self.p_v = p_v
        self.p_i = p_i
        self.n_v = n_v
        self.n_i = n_i

    def _get_laws(self, s):
        return [EquationLaw(s.p_i + s.n_i, 0)]


class ResistorElement(TwoPinElement):
    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        R: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.R = R

    def _get_laws(self, s):
        return [*super()._get_laws(s), EquationLaw(s.p_v - s.n_v, s.R * s.p_i)]


class CapacitorElement(TwoPinElement):
    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        n_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        u: Annotated[float, StateSpec()] = 0.0,
        C: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, p_v=p_v, p_i=p_i, n_v=n_v, n_i=n_i)
        self.u = u
        self.C = C

    def _get_laws(self, s):
        return [*super()._get_laws(s), EquationLaw(s.C * derivative(s.u, 1), s.p_i), EquationLaw(s.u, s.p_v - s.n_v)]


class GroundElement(Element):
    ports: ClassVar = {'p': ELECTRICAL}

    def __init__(
        self,
        label,
        *,
        p_v: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        p_i: Annotated[float, StateSpec(algebraic=True)] = 0.0,
    ):
        super().__init__(label)
        self.p_v = p_v
        self.p_i = p_i

    def _get_laws(self, s):
        return [EquationLaw(s.p_v, 0)]


circuit = System(
    elements=[CapacitorElement('tank', u=1.0, C=0.5), ResistorElement('load', R=2.0), GroundElement('ground')],
    interactions=[
        ConnectionInteraction('top', members=['tank', 'load'], ports=['p', 'p']),
        ConnectionInteraction('bottom', members=['tank', 'load', 'ground'], ports=['n', 'n', 'p']),
    ],
)
trajectory = Simulation(circuit, dt=0.1).prepare().run(until=1.0).trajectory_
assert np.isclose(trajectory.values('tank.u')[-1], np.exp(-1.0), rtol=1e-3)
assert np.allclose(trajectory.values('load.p_i').ravel(), trajectory.values('tank.u').ravel() / 2.0)
trajectory.values('tank.u')[-1].round(4)
```

The voltage of the capacitor decays as `exp(-t / RC)`, and the current through the load is that voltage over `R`. The
systems of the field `engineering` in the catalog hold electrical and thermal components built in the same way. A port
component drives a mechanism through an `EquationLaw` whose target is a coordinate of an `EulerLagrangeInteraction`,
which adds to the equation of that coordinate, as `dc_motor` shows.

#### The solve

`OrdersPass` solves the equations of a system with algebraic states inside the right side of its differential
equations, so its solvers run unchanged.

- The unknowns are the highest derivatives and the algebraic states.
- A maximum bipartite matching pairs the equations with the unknowns. The strongly connected components of their
  dependency graph, in topological order, are the blocks.
- A block linear in its unknowns is solved symbolically up to six unknowns, and by a numeric linear solve above.
- Any other block runs `scipy.optimize.root` on the NumPy backend, with the Jacobian Sage derives, at every evaluation,
  from the values of the last evaluation. When that fails it starts again one unit away, and then raises `SolverError`.
- The laws that read an algebraic state read its solution. A function argument, an event guard, or a reset that reads
  an algebraic state of a nonlinear block raises `ValidationError`, because these need a symbolic value.

The solve has no index reduction, so the structural check below reports a system of a higher index.

#### Issues of connected systems

`PassCompiler` reports these issues with the other issues of a system, before compilation.

- `port-connector`: the ports of one connection have different connectors, or a connection names a port that no single
  element declares. Connect declared ports of one connector.
- `port-connections`: a port appears in more than one connection. List the ports of the node in one connection.
- `port-states`: an element type declares a port without an algebraic state for one of its variables. Declare
  `<port>_<variable>` with `StateSpec(algebraic=True)`.
- `structural-singularity`: the matching leaves equations or unknowns unmatched, as for two capacitors in parallel. The
  issue names the constrained states. Add a compliant element, such as a resistance, between them.
- `algebraic-law`: a law other than an `EquationLaw` targets an algebraic state. State its value with an `EquationLaw`.

A connection under discrete time gives the `law-time` issue of every `EquationLaw`.

### Delays, events, drivers, and noise

Make laws read the past, change states at discrete events, follow inputs that switch in time, and take noise in the
convention of your field.

#### Delays

Read a state as it was some time ago, as in a feedback that acts with a lag. `delayed(x, tau)` from
`skcomplex.core.components.laws.utils` is the state `x` at `t - tau`.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import delayed
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


def wavy_history(t):
    return 0.5 + 0.1 * np.sin(t)


class MackeyGlassElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec(history=wavy_history)] = 0.5,
        beta: Annotated[float, ParameterSpec()] = 0.2,
        gamma: Annotated[float, ParameterSpec()] = 0.1,
        n: Annotated[float, ParameterSpec()] = 10.0,
        tau: Annotated[float, ParameterSpec()] = 17.0,
    ):
        super().__init__(label)
        self.x = x
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.tau = tau

    def _get_laws(self, s):
        past = delayed(s.x, s.tau)
        return [RateLaw(s.x, s.beta * past / (1 + past**s.n) - s.gamma * s.x)]


cells = System(elements=[MackeyGlassElement('cells')])
round(float(Simulation(cells, dt=0.5).prepare().run(until=300).trajectory_.values('cells.x')[-1]), 3)
```

- The delay is a constant or depends on the state.
- The solvers keep a history of dense output and step no further than the smallest delay, by the method of steps.
- Before the start, the history is the initial value, or what the state declares with `StateSpec(history=...)`, a
  number or a function of time.

#### Events

Change states at the moment a condition is met, such as a ball that bounces off the floor.
`CrossingLaw(guard, reset, direction, terminal, label='')` fires when the guard crosses zero.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import CrossingLaw, RateLaw
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class BallElement(Element):
    def __init__(
        self,
        label,
        *,
        h: Annotated[float, StateSpec(order=2)] = 1.0,
        g: Annotated[float, ParameterSpec()] = 9.81,
        e: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label)
        self.h = h
        self.g = g
        self.e = e

    def _get_laws(self, s):
        bounce = {derivative(s.h, 1): -s.e * derivative(s.h, 1)}
        return [RateLaw(s.h, -s.g), CrossingLaw(s.h, bounce, direction=-1, label='impact')]


trajectory = Simulation(System(elements=[BallElement('ball')]), dt=0.01).prepare().run(until=3).trajectory_
[round(event.time, 6) for event in trajectory.events]
```

- The guard fires upward for `direction=1`, downward for `-1`, and either way for `0`.
- The solver locates the crossing on its interpolant to the precision of the root finder. It applies the reset to every
  named state at once, and restarts from there.
- A terminal event ends the run at the crossing.
- Every frame carries the events since the previous one. The trajectory keeps them all, with their times and the states
  before and after the reset.
- A guard that fires again right after its reset raises an error that names the chattering event.

#### Drivers

Feed a system an input that switches or bends in time, such as a step, a ramp, or measured data. `step`, `ramp`,
`pulse`, `pulse_train`, and `series` from `skcomplex.core.components.laws.utils` are inputs of the time `s.t`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import step
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class FilterElement(Element):
    def __init__(
        self,
        label,
        *,
        y: Annotated[float, StateSpec()] = 0.0,
        k: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.y = y
        self.k = k

    def _get_laws(self, s):
        return [RateLaw(s.y, -s.k * s.y + step(s.t, 5.0))]


filtered = Simulation(System(elements=[FilterElement('filter')]), dt=0.5).prepare().run(until=10).trajectory_
round(float(filtered.values('filter.y')[-1]), 4)
```

`series` interpolates data with `'linear'` or `'previous'` interpolation. The compiler finds the times where the inputs
jump or bend, including the periodic edges of pulse trains. The solvers stop at every one and evaluate the inputs on the
correct side of it, so a step response is exact to the solver tolerance.

#### Noise conventions

Read noise in the convention of your field, Itô, Stratonovich, or anti-Itô.
`NoiseLaw(..., convention='stratonovich')` or `'anti-ito'` declares how a noise law is read.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import MilsteinSolver
from skcomplex.core.specs import ParameterSpec, StateSpec


class GrowthElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        mu: Annotated[float, ParameterSpec()] = 0.5,
        sigma: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label)
        self.x = x
        self.mu = mu
        self.sigma = sigma

    def _get_laws(self, s):
        return [RateLaw(s.x, s.mu * s.x), NoiseLaw(s.x, s.sigma * s.x, convention='stratonovich')]


growth = System(elements=[GrowthElement('asset')])
round(float(Simulation(growth, solver=MilsteinSolver(dt=0.01), random_state=0).prepare().run(until=1).state_[0]), 3)
```

The compiler adds the drift correction `α g ∂g/∂x`, with `α = 1/2` for Stratonovich and `1` for anti-Itô. So every SDE
solver integrates the equivalent Itô equation.

- `EulerMaruyamaSolver` has strong order 1/2.
- `MilsteinSolver`, in Platen's derivative-free form, has strong order 1 for diagonal noise.
- `HeunSolver` averages the drift over every step.

#### Shared and correlated noise

Drive several states with one source of noise, or with sources that move together, such as the prices of two related
assets. `NoiseLaw(..., process=...)` names the Wiener process a law draws.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import NoiseLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.specs import RoleSpec, StateSpec, WienerProcessSpec


class StockElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.n = n


class CorrelatedNoiseInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        noise: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
    ):
        super().__init__(label)
        self.first = first
        self.second = second
        self.noise = noise

    def _get_laws(self, s):
        return [NoiseLaw(s.first.n, 1.0, process=s.noise.dW[0]), NoiseLaw(s.second.n, 1.0, process=s.noise.dW[1])]


processes = WienerProcessSpec('market', dimension=2, covariance=[[1.0, 0.8], [0.8, 1.0]])
market = System(
    elements=[StockElement('a'), StockElement('b')],
    interactions=[CorrelatedNoiseInteraction('drive', first='a', second='b', noise=processes)],
)
solver = EulerMaruyamaSolver(dt=1.0)
trajectory = Simulation(market, solver=solver, dt=1.0, random_state=0).prepare().run(until=2000).trajectory_
round(float(np.corrcoef(np.diff(trajectory.values('a.n')), np.diff(trajectory.values('b.n')))[0, 1]), 1)
```

The measured correlation of the increments is the 0.8 of the covariance.

- Without a process, every element and every group member takes its own independent process.
- Laws of one element, or of one member, that name the same label share one increment. So a reaction can drive every
  species it changes with one process.
- A `WienerProcessSpec(label, dimension, covariance)` from `skcomplex.core.specs` declares processes whose increments
  over a step `h` have the covariance `h C`.
- A `WienerProcessSpec` is no element. An interaction binds it through a role annotated with `RoleSpec(noise=True)`, and
  its noise laws name process `k` with `s.<role>.dW[k]`, which every element, and every member of a group, can share.
- Interactions of one system that bind processes of one label share them and their correlations.
- The covariance is fixed when the system compiles. NumPy factors it once, so the steppers scale standard normal
  increments drawn from the run's random generator.

The Stratonovich and anti-Itô corrections of each law use the full diffusion matrix and the covariance. So a state that
several laws drive, and laws that share a process, are corrected exactly. Laws that share a process declare one
convention. Validation reports:

- laws of different conventions on one process,
- a Stratonovich or anti-Itô amplitude that reads a state another element drives through its process,
- a covariance that is not symmetric positive semidefinite,
- a handle `dW` used anywhere but as the process of a noise law.

#### Conserved noise

Add thermal noise to a field without changing its total, as in System B and active field theories.
`NoiseLaw(target, value, conserved=True)` on a state of a field on a lattice adds the divergence of a random flux,
`div(value eta)`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import NoiseLaw, RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerMaruyamaSolver
from skcomplex.core.specs import ParameterSpec, StateSpec


class DensityElement(Element):
    def __init__(
        self,
        label,
        *,
        rho: Annotated[float, StateSpec()] = 1.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
        T: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label)
        self.rho = rho
        self.D = D
        self.T = T

    def _get_laws(self, s):
        return [RateLaw(s.rho, s.D * laplacian(s.rho)), NoiseLaw(s.rho, sqrt(2 * s.T), conserved=True)]


gas = System(elements=[FieldElement('gas', element=DensityElement, space=LatticeSpace((16, 16)))])
solver = EulerMaruyamaSolver(dt=0.01)
density = Simulation(gas, solver=solver, random_state=0).prepare().run(until=5).trajectory_.values('gas.rho')
round(float(density[-1].sum()), 6), bool(density[-1].std() > 0)
```

The density fluctuates, and its total stays at 256, one per site.

- Every link between neighbouring sites draws one Wiener process. Site `i` gains `value / h` times the increment of the
  link from its previous neighbour and loses the same amount on the link to its next one, along every dimension of
  spacing `h`.
- The total of the field then changes only by its rate laws, exactly to rounding.
- The links that leave a lattice through a zero-flux or fixed edge carry no process.
- The value is a number or an expression of parameters, the same on every link, and the field uses finite differences
  on a square lattice.
- Validation reports a conserved noise law on any other state with the code `noise-unsupported`.

#### Which solvers support what

Solvers and backends declare the noise structures they draw.

- `'diagonal'` is one process per state.
- `'shared'` is a process that drives several states.
- `'correlated'` is the processes of a `WienerProcessSpec` with a covariance.
- `'conserved'` is the link processes of conserved noise.

`EulerMaruyamaSolver`, `HeunSolver`, and the NumPy and Numba backends support all four. `MilsteinSolver`, whose
derivative-free correction holds for diagonal noise only, and `LammpsSolver` support diagonal noise. Validation
reports a system with shared, correlated, or conserved noise for them before compiling it. The compatibility page lists
the declarations.

SciPy, RK4, Euler-Maruyama, Milstein, and Heun handle delays, events, and drivers. Velocity Verlet, Gillespie, and map
iteration reject delays and events, with an error that names the solvers to use.

#### Ready-made systems with delays and events

The catalog has these systems ready.

- Delays, events, and drivers: `mackey_glass` in physiology, `hutchinson` in ecology, `integrate_and_fire` and
  `izhikevich` in neuroscience, `delayed_kuramoto` in synchronization, `stochastic_resonance` and `brownian_near_wall`
  in soft matter, and `bouncing_ball`, `newtons_cradle`, and `circular_billiard` in mechanics.
- Noise: `BindingInteraction` and `DissociationInteraction` take `langevin=True` for chemical Langevin equations, as in
  `chemical_langevin_network`. `correlated_geometric_brownian_motion` drives two assets with correlated processes, and
  `active_system_b` and `cahn_hilliard` with a `temperature` take conserved noise.

### Stochastic maps

Write a discrete-time stochastic process, such as a random walk or a Markov chain, with update laws that draw random
numbers. The draws come from the simulation's random generator, so two runs with the same `random_state` give the same
trajectory.

#### Random draws

Use a draw in an update law like any other expression. `uniform(name, low, high)`, `normal(name, mean, deviation)`,
`bernoulli(name, p)`, and `categorical(name, weights)` are symbolic random values.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, normal, where
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class PersistentElement(Element):
    def __init__(
        self,
        label,
        *,
        d: Annotated[float, StateSpec()] = 1.0,
        x: Annotated[float, StateSpec()] = 0.0,
        q: Annotated[float, ParameterSpec()] = 0.8,
    ):
        super().__init__(label)
        self.d = d
        self.x = x
        self.q = q

    def _get_laws(self, s):
        turn = where(bernoulli('turn', 1 - s.q) > 0, -s.d, s.d)
        return [UpdateLaw(s.d, turn), UpdateLaw(s.x, s.x + turn + normal('jitter', 0, 0.1))]


walkers = System(elements=[GroupElement('walkers', element=PersistentElement, size=500)])
final = Simulation(walkers, dt=1.0, random_state=0).prepare().run(until=200).trajectory_.values('walkers.x')[-1]
round(float(final.var() / 200), 1)
```

Each walker keeps its direction with probability `q` and turns back otherwise, with a small jitter on every step.

- A draw takes a fresh value at every step, and every member of a group draws its own value.
- The arguments of a draw can be expressions of states and parameters, and they take the values from before the step.
- Every draw carries a name. Every use of the same name in one element reads the same value within a step, and draws
  with different names are independent.
- Two uses of one name with different distributions or arguments fail to compile.
- Within a step, the draws are taken by element path, then draw name, then member index. So the trajectory does not
  depend on the order of the laws.
- The NumPy and Numba backends give equal trajectories for uniform, Bernoulli, and categorical draws.

Only update laws and renewal laws take draws. A draw in a rate, noise, jump, boundary, equation, or crossing law raises
`IncompatibleLawsError`, since noise laws and jump laws give randomness in continuous time. Draws are not states, so the
trajectory does not record them. A system that wants to keep a draw assigns it to a state.

#### Markov chains and hidden Markov sources

Step a state over a finite alphabet with a transition matrix, and emit an observed symbol from a hidden state.
`TransitionLaw(targets, matrix)` is the step of a discrete-time Markov chain, and `EmissionLaw(targets, source, matrix)`
draws an observed state.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EmissionLaw, TransitionLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class SourceElement(Element):
    def __init__(
        self,
        label,
        *,
        hidden: Annotated[str, StateSpec(domain=FiniteEnumeratedSet(['calm', 'storm']))] = 'calm',
        symbol: Annotated[str, StateSpec(domain=FiniteEnumeratedSet(['a', 'b']))] = 'a',
        stay: Annotated[float, ParameterSpec()] = 0.9,
    ):
        super().__init__(label)
        self.hidden = hidden
        self.symbol = symbol
        self.stay = stay

    def _get_laws(self, s):
        return [
            TransitionLaw(s.hidden, [[s.stay, 1 - s.stay], [0.4, 0.6]]),
            EmissionLaw(s.symbol, s.hidden, [[0.7, 0.3], [0.2, 0.8]]),
        ]


simulation = Simulation(System(elements=[SourceElement('source')]), dt=1.0, random_state=0)
simulation.prepare().run(until=10000)
round(float((simulation.trajectory_.values('source.hidden') == 'calm').mean()), 1)
simulation.update_parameters({'source.stay': 0.5})
```

- The target of a `TransitionLaw` is a state over a finite alphabet, a `FiniteEnumeratedSet` of labels or the integers
  from 0 to `n - 1`. Its current value selects the row of the transition matrix that the next value is drawn from.
- An `EmissionLaw` draws the observed state from the row of an emission matrix that the hidden state `source` selects.
  It reads the hidden state after the same step, so the two sequences align frame by frame.
- Both are update laws, so they combine with other update laws.
- Matrix entries are numbers or expressions of parameters. Compiling checks that the entries are nonnegative and that
  every row sums to 1.
- `update_parameters` checks again every matrix a changed parameter enters. It rejects a change that breaks a matrix
  with a `ParameterError`, before the change applies.

#### Ready-made stochastic maps

The catalog has these systems ready: `persistent_random_walk`, `markov_source`, `hidden_markov_source`, and
`random_walk_with_memory` in stochastic processes, `chaos_game` in fractals, `probabilistic_boolean_network` and
`intrinsic_instability` in biology, and `diluted_hopfield` in neural networks.

### Anomalous processes

Write processes with memory: heavy-tailed jumps, events that fire after waiting times from any distribution, noise
whose increments are correlated over the whole run, and derivatives of fractional order.

#### Draws from SciPy distributions

Draw from any distribution of `scipy.stats`, such as a Lévy stable law for long flights.
`draw(name, distribution, **parameters)` takes the name of the distribution, such as `'pareto'`, `'lomax'`, `'expon'`,
`'beta'`, or `'levy_stable'`, with its shape parameters, `loc`, and `scale` as keywords.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import draw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class FlierElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [UpdateLaw(s.x, s.x + draw('jump', 'levy_stable', alpha=1.5, beta=0.0))]


fliers = System(elements=[GroupElement('fliers', element=FlierElement, size=100)])
flight = Simulation(fliers, dt=1.0, random_state=0).prepare().run(until=20)
flight.trajectory_.values('fliers.x').shape
```

- The keywords can be expressions of states and parameters.
- A SciPy draw follows the rules of the other named draws, and it samples with the distribution's `rvs` and the
  simulation's random generator.
- Compiling rejects a name `scipy.stats` does not provide, listing the closest names, and keywords the distribution does
  not take.
- In update laws, a SciPy draw is sampled for every member before each step, and the map then steps in Python with the
  backend's compiled update.

#### Jumps on clocks of waiting times

Fire an event after a random waiting time, such as a trapped particle that escapes after a heavy-tailed wait.
`JumpLaw(waiting=..., reset=..., label=...)` fires after a waiting time, applies its reset to the states all at once,
and draws the next waiting time from the states after the reset.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.laws.utils import draw, normal
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class WalkerElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        alpha: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.x = x
        self.alpha = alpha

    def _get_laws(self, s):
        waiting = draw('wait', 'pareto', b=s.alpha)
        return [JumpLaw(waiting=waiting, reset={s.x: s.x + normal('jump', 0, 1)}, label='jumps')]


walkers = System(elements=[GroupElement('walkers', element=WalkerElement, size=1000)])
walk = Simulation(walkers, dt=100.0, random_state=0).prepare().run(until=1000)
walk.problem_.form.value, bool(walk.trajectory_.values('jumps.renewals')[-1] > 1000)
```

- Every member of a group renews on its own clock.
- Draws in the waiting time and the reset take a fresh value at every renewal of every member, and every use of one
  name in a renewal reads the same value.
- A waiting time that is a single draw must come from a distribution with nonnegative support. Compiling rejects one
  that can be negative, such as `normal`.

A system or partition whose only laws are such jumps compiles to the renewal form. `RenewalSolver` runs it exactly, from
renewal to renewal with no time grid, and a frame holds the states the last renewal at or before its time set.

These jumps also combine with the rate and noise laws of the same element, which evolve the states between renewals.
`RK4Solver` and `EulerMaruyamaSolver` stop every member at its own renewals when the members do not interact. When a
law sums over the group, they split the step at every renewal time for all members. Other steppers reject these jumps.

The trajectory records every renewal pass as a `RenewalRecord` among its events, with the time and the member of every
renewal and the member's states before and after it. The observable `<label>.renewals` counts the renewals up to every
frame. The Rerun observer streams it as a time series, and the renewals as text entries under `renewals/<label>`. The
compiler keeps the count of every member in a hidden state, `<group>.<label>~renewals`.

#### Fractional noise

Drive a state with noise that remembers its past, as in rough or persistent paths.
`FractionalNoiseLaw(targets, value, hurst=hurst)` drives a state with fractional Brownian motion of Hurst index
`0 < H < 1`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import FractionalNoiseLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import FractionalEulerMaruyamaSolver
from skcomplex.core.specs import StateSpec


class RoughElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [FractionalNoiseLaw(s.x, 1, hurst=0.3)]


rough = Simulation(
    System(elements=[GroupElement('walkers', element=RoughElement, size=500)]),
    solver=FractionalEulerMaruyamaSolver(dt=0.01),
    dt=0.1,
    random_state=0,
).prepare().run(until=1)
round(float(rough.trajectory_.values('walkers.x')[-1].var()), 1)
```

- The increments anticorrelate for `H < 1/2` and correlate for `H > 1/2`.
- A Hurst index of 1/2 compiles to an ordinary noise law. For other indices the amplitude must not depend on any state.
- `FractionalEulerMaruyamaSolver(dt, horizon)` is the stepper of systems with fractional noise. When it first advances,
  it draws the increments of the whole run on its grid, with the circulant embedding of Davies and Harte over
  `scipy.fft`. So frames must fall on multiples of `dt`.
- A simulation passes the end of its run as the horizon, and advancing beyond it raises `HorizonError`.
- Other SDE steppers reject fractional noise with an error that names `FractionalEulerMaruyamaSolver`.
- `fractional_gaussian_noise` returns the noise paths themselves.

The catalog has these systems ready: `continuous_time_random_walk`, `levy_walk`, `annealed_transient_time_motion`,
`fractional_brownian_motion`, and `levy_flight` in soft matter, and `trap_system` in statistical mechanics.

#### Fractional derivatives

Give a state a derivative of fractional order, for relaxation that is slower than any exponential. A state declared
with a fractional order, `StateSpec(order=0.6)`, has a Caputo derivative of that order.

```python
from typing import Annotated

from skcomplex.core.analysis import SymbolicAnalysis
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import L1Solver
from skcomplex.core.specs import StateSpec


class RelaxingElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec(order=0.6)] = 1.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x)]


system = System(elements=[RelaxingElement('a')])
relaxing = Simulation(system, solver=L1Solver(dt=0.01, memory=10.0), dt=1.0).prepare().run(until=20)
solution = SymbolicAnalysis('solve', use_initial_state=True).prepare(system).run().solutions_['a.x'].rhs()
exact = solution.subs({solution.variables()[0]: 20.0})
relaxing.problem_.form.value, abs(float(relaxing.trajectory_.values('a.x')[-1]) - float(exact)) < 1e-3
```

The simulated relaxation matches the exact Mittag-Leffler solution to `1e-3` at time 20.

The rate laws on a fractional state give its derivative, `D^alpha x = f(t, x)`, and superpose as for any state, so no
other law kind is needed. Order 1 is the ordinary rate law. The derivative of order `alpha` at a time depends on the
whole past of the state, so the solution relaxes like a Mittag-Leffler function, slower than any exponential.

A system with a state of a fractional order compiles to the fractional form, with the order of every state component
in `problem.orders`. The states of order 1 of the same system keep their ordinary equations. Three steppers integrate
the fractional form on a fixed grid from the start of the run, so frames must fall on multiples of their `dt`.

- `GrunwaldLetnikovSolver(dt, memory=None)` weighs the past states with the binomial weights
  `(-1)^j binomial(alpha, j)`.
- `L1Solver(dt, memory=None)` interpolates the state linearly between grid points inside the Caputo integral.
- `FractionalABMSolver(dt)` is the predictor-corrector of Diethelm, Ford, and Freed, of order `1 + alpha` for smooth
  solutions, and the default.

The two explicit schemes are of order 1. All three keep the whole history, so a run of `n` steps costs `n^2`
operations. `memory` bounds that cost. The history of the last `memory` time units enters with its exact weights, and
the older history enters through about 40 exponential modes fitted to the weights, so every step costs the same. For
fractional relaxation of order 0.6, a window of 10 time units stays within `1e-8` of the whole history at time 20. A
change of a fractional state between frames, such as one the simulation's interventions make, starts a new history from
the changed state.

Validation reports the misuses of fractional orders before compiling, each with the path of the state and a fix.

- `fractional-order`: an order that is neither an integer nor in `(0, 1)`.
- `fractional-time`: a fractional state in a system whose declared time is not continuous.
- `order-unsupported`: a chosen solver or backend that does not integrate fractional orders.
- `solver-unsupported`: a law on a fractional state that no fractional solver runs, such as a noise law.

Solvers and backends declare the kinds of order they integrate in the `orders` field of their `SupportSpec`,
`'integer'` or `'fractional'`. The NumPy and Numba backends run the fractional form.

`SymbolicAnalysis('solve')` solves one linear fractional relaxation `D^alpha x = c x` as `x(0) E_alpha(c t^alpha)`. It
writes the solution with the symbolic function `mittag_leffler`, which evaluates numerically. It sums the power series
in arbitrary precision with Sage for moderate arguments, and takes the asymptotic expansion for large ones.
`SymbolicAnalysis('equations')` writes a Caputo derivative as `caputo(x(t), t, alpha)`.

The order belongs to the element type, so a sweep over orders builds one system per order. The catalog has these
systems ready in fractional dynamics, each with the order as an argument: `fractional_relaxation`,
`fractional_oscillator`, `fractional_van_der_pol`, and `fractional_lorenz`.

### Geometry of chains

Measure the shape of a chain of points, such as a polymer or a protein backbone, frame by frame. `ShapeAnalysis` reads
frames by points by dimensions, from an array or a trajectory.

```python
import numpy as np

from skcomplex.core.analysis import ShapeAnalysis

chain = np.array([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]])
assert ShapeAnalysis('end-to-end').prepare(chain).run().values_.tolist() == [5.0]
ShapeAnalysis('end-to-end').prepare(chain).run().values_
```

`ShapeAnalysis(kind, group=None, reference=None)` takes one of three kinds.

- `'gyration'` gives the radius of gyration per frame.
- `'end-to-end'` gives the distance between the first and the last point.
- `'rmsd'` gives the root mean square deviation from `reference` after optimal superposition.

### Groups of many elements

Use a group when your system has many identical elements, such as the molecules of a gas, the cars on a road, or the
neurons of a network. You write the laws once for one member, and they run over all members at array speed.

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

rng = np.random.default_rng(0)
values = {'x': rng.uniform(0, 1, (1_000, 2)), 'x_d1': rng.normal(0, 1, (1_000, 2))}
gas = System(
    elements=[GroupElement('gas', element=PointElement, size=1_000, values=values)],
    space=ContinuousSpace(extent=(1.0, 1.0), boundary='reflecting'),
)
trajectory = Simulation(gas, dt=0.01).prepare().run(until=0.5).trajectory_
trajectory.values('gas.x').shape
```

A group holds N identical elements of one type as arrays, with one row per member. It compiles once, whatever its
size, and every law written for one member runs over all members with array operations.

- A group is `GroupElement(label, element=..., size=..., values={...})`, with per-member values by state name. The
  states and parameters of agents cannot take the names of the agent view, such as `location` or `alive`.
- A value is an array whose first dimension is the size of the group, or one value shared by every member.
- An interaction whose role binds to a group applies to every member. A single element bound next to the group
  receives the sum over the members, unless its law reads the group only through reductions.
- Group variables are named by the group path, such as `gas.x` with shape `(1000, 2)`.
- A group parameter takes one value for every member, or one value per member, in `update_parameters`.

The catalog system `ideal_gas` builds such a gas of non-interacting molecules in a reflecting box, with the parameters
`n_molecules`, `kT`, and `size`:

```python
from skcomplex.systems import create_system

create_system('ideal_gas', n_molecules=200, kT=2.0, size=1.0).elements[0].size
```

#### Reductions over a group

Use a reduction when a member's law needs a quantity of the whole group, such as its total, its mean, or the number of
members above a threshold. Here every unit reads the total of the group in its own law:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
Simulation(units, dt=0.5).prepare().run(until=2).trajectory_.values('units.x')[-1, :3].round(3).tolist()
```

`total`, `mean`, `maximum`, `minimum`, `variance`, and `count` of `skcomplex.core.components.laws.utils` reduce an
expression over the members of a group inside any law. Inside the reduction, the symbols of a member stand for the
reduced member. Outside it, they stand for the reading member. So `RateLaw(s.x, -s.x + tanh(total(s.x)))` reads the
group total in every member's own law.

A reduction is evaluated once per evaluation, before the member laws, at a cost linear in the size of the group, on
NumPy and Numba.

Reductions also take weights and nest. Replicator dynamics, where each strategy grows by how much its payoff beats the
mean payoff, needs both:

```python continuation
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import RoleSpec

payoff = np.array([[-0.5, 2.0], [0.0, 1.0]])


class StrategyElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.x = x


class ReplicationInteraction(Interaction):
    def __init__(self, label, *, population: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.population = population

    def _get_laws(self, s):
        x = s.population.x
        return [RateLaw(x, x * (total(x, weights=payoff) - total(x * total(x, weights=payoff))))]


game = System(
    elements=[GroupElement('population', element=StrategyElement, size=2, values={'x': [0.1, 0.9]})],
    interactions=[ReplicationInteraction('replication', population='population')],
)
Simulation(game, dt=1.0).prepare().run(until=40).trajectory_.values('population.x')[-1].round(3).tolist()
```

The arguments of a reduction are:

- `where` selects the members with a condition on the reduced member, such as `count(where=s.x > 0.5)`. A reduction
  that selects no member is 0.
- `weights` gives `total` and `mean` a constant matrix, a NumPy array or a SciPy sparse matrix, with one row per
  reading member and one column per reduced member. Member `i` then reads `sum_j W_ij a_j`, such as a payoff matrix.
  A vector with one weight per reduced member is a row that every reading member shares, such as the activities of a
  moment. A shaped parameter, such as `weights=s.W`, gives weights that a schedule, an experiment, or a training
  interaction changes, as [Shaped parameters and coupling tensors](#shaped-parameters-and-coupling-tensors) shows.
- `over` names the group by its namespace, such as `over=s.neurons`, when the expression reads no symbol of it.

Reductions follow a few rules:

- A reduction reads one group, and an argument that reads two groups raises an error.
- Reductions nest, so the mean fitness `total(s.x * total(s.x, weights=A))` of replicator dynamics evaluates the inner
  reduction first.
- An interaction reads groups member by member outside reductions, which needs groups of one size. It reads other
  groups through reductions.
- A law that sets a single element from reductions alone runs once, not once per member.
- Discrete maps read the reductions at step `n`, jump propensities after every event, and events at the state they
  receive.

**Element sets.** A role also binds to an element set, a list of single elements of one type. The interaction's laws
apply to each element, the elements keep their own state names, and reductions over the role reduce over the set.
Here three named units inhibit each other:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class ActivityElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.1):
        super().__init__(label)
        self.x = x


class InhibitionInteraction(Interaction):
    def __init__(self, label, *, units: Annotated[str, RoleSpec()], beta: Annotated[float, ParameterSpec()] = 2.0):
        super().__init__(label)
        self.units = units
        self.beta = beta

    def _get_laws(self, s):
        x = s.units.x
        return [RateLaw(x, x * (1 - x) - s.beta * x * (total(x) - x))]


network = System(
    elements=[ActivityElement(f'unit_{index}', x=0.1 * (index + 1)) for index in range(3)],
    interactions=[InhibitionInteraction('inhibition', units=['unit_0', 'unit_1', 'unit_2'])],
)
(Simulation(network, dt=1.0).prepare().run(until=30).state_.round(3) + 0.0).tolist()
```

#### Variadic roles

Use a variadic role when one law takes any number of inputs, such as a node that sums weighted signals from a chosen
list of other nodes:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class NodeElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class WeightedSumInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[list[str], RoleSpec(variadic=True)],
        weights: Annotated[list[float], ParameterSpec(per='inputs')] = 1.0,
    ):
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.weights = weights

    def _get_laws(self, s):
        total_input = sum(weight * item.x for weight, item in zip(s.weights, s.inputs, strict=True))
        return [UpdateLaw(s.target.x, total_input)]


summed = System(
    elements=[
        NodeElement('a', x=1.0),
        NodeElement('b', x=2.0),
        NodeElement('out'),
        GroupElement('nodes', element=NodeElement, size=5, values={'x': [0.0, 1.0, 2.0, 3.0, 4.0]}),
    ],
    interactions=[WeightedSumInteraction('sum', target='out', inputs=['a', 'b', 'nodes[3]'], weights=[1.0, 1.0, 10.0])],
)
simulation = Simulation(summed, dt=1).prepare().run(until=1)
simulation.update_parameters({'sum.weights[2]': 0.0})
simulation.trajectory_.values('out.x').tolist(), simulation.parameters_.tolist()
```

- A role declared with `RoleSpec(variadic=True)` binds a list of element paths and group member references
  `<group>[<i>]`. Laws read it as a tuple of namespaces in the given order.
- A parameter declared with `ParameterSpec(per='<role>')` holds one value per entry, and laws read it as a tuple of
  symbols. Each value is the parameter component `<path>.<name>[<i>]`, which `update_parameters` and `schedule` set
  like any parameter.

An element set and a variadic role both bind a list. An element set applies the laws once per element, each element
read like a single element. A variadic role gives one law the whole sequence. `random_boolean_network` and
`yeast_cell_cycle` read the inputs of every node through one variadic role.

A role declared with `RoleSpec(noise=True)` binds a `WienerProcessSpec` instead of a path, as the shared and
correlated noise of [Delays, events, drivers, and noise](#delays-events-drivers-and-noise) shows.

#### Several groups and member offsets

Use a member offset when a member reads its neighbour in the group order, such as a car that reads the car ahead on a
ring road. A small push to the first car grows into a traffic jam:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import mod, shift_member
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class CarElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0, v: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.x = x
        self.v = v

    def _get_laws(self, s):
        return [RateLaw(s.x, s.v)]


class FollowingInteraction(Interaction):
    def __init__(self, label, *, cars: Annotated[str, RoleSpec()], length: Annotated[float, ParameterSpec()] = 40.0):
        super().__init__(label)
        self.cars = cars
        self.length = length

    def _get_laws(self, s):
        headway = mod(shift_member(s.cars.x, 1) - s.cars.x, s.length)
        return [RateLaw(s.cars.v, tanh(headway - 2.0) + tanh(2.0) - s.cars.v)]


positions = np.arange(20) * 2.0 + np.eye(20)[0] * 0.1
ring = System(
    elements=[GroupElement('cars', element=CarElement, size=20, values={'x': positions})],
    interactions=[FollowingInteraction('following', cars='cars', length=40.0)],
)
bool(Simulation(ring, dt=1.0).prepare().run(until=50).trajectory_.values('cars.v')[-1].std() > 0.1)
```

`shift_member(expression, offset, boundary='periodic', fill=0.0)` reads an expression of a group's members at the
member `offset` places along the group order, the member index.

- Past the ends, `'periodic'` wraps around, `'reflecting'` mirrors the order, and `'fixed'` reads `fill`.
- Nested offsets add, and an offset read of a single element raises an error.
- The read compiles to an index table of the group, so one kernel runs over every member on the NumPy and the Numba
  backends.

The ring roads of `optimal_velocity` and `intelligent_driver_system` read the car ahead this way, with `mod` for the
headway across the point where positions wrap.

Roles of one interaction can also bind several groups of one size, and the laws read row `i` of every group together.
Groups of different sizes raise a `RoleError`, since coupling every member of one group with every member of another
is the work of a pair interaction or a reduction.

#### Jacobians of reductions

Implicit solvers need the Jacobian of the system, and the library computes it exactly, even through reductions:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import tanh

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.specs import RunStateSpec, StateSpec


class UnitElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, s):
        return [RateLaw(s.x, -s.x + tanh(total(s.x)))]


units = System(elements=[GroupElement('units', element=UnitElement, size=50, values={'x': np.linspace(-1, 1, 50)})])
problem = compile_system(units)
state = RunStateSpec(0.0, problem.initial_state, problem.initial_parameters)
jacobian = problem.jacobian.prepare(problem).run(state).values_
jacobian.shape, problem.jacobian.pattern().nnz
```

A reduction makes every reading member depend on every reduced member. The Jacobian of a drift with reductions
therefore has the rank-one block `dF_i/dR dR/dx_j` of every reduction `R`.

- `problem.jacobian` of an ODE problem is a kernel. Prepared with the problem and run at a `RunStateSpec`, it sets in
  `values_` the exact Jacobian, by the chain rule through every reduction, as a SciPy sparse matrix.
- `problem.jacobian.pattern()` states its structure, which the implicit methods of `ScipyODESolver` receive (see
  [Discretizations and stiff solvers](#discretizations-and-stiff-solvers)).
- When the dense blocks exceed 4 million entries, `pattern()` returns `None` and an implicit solver estimates a dense
  Jacobian.

Mean-field groups are rarely stiff, so an explicit method such as `ScipyODESolver(method='RK45')` suits large groups
with reductions.

`StabilityAnalysis` and the other analyses of the vector field in `skcomplex.core.analysis` expand a group of at most
64 members into one equation per member, with every reduction written as an explicit sum. The fixed points of
replicator dynamics are therefore derived symbolically. A larger group raises an error that states the limit.

#### Pair interactions

Use a pair interaction for forces between members that depend on their distance, such as the atoms of a liquid:

```python
import numpy as np

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.interactions import PairCouplingInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation


def lennard_jones(s, r):
    return 4 * (r**-12 - r**-6)


side = np.arange(10) * 1.2
positions = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
liquid = System(
    elements=[GroupElement('atoms', element=PointElement, size=100, values={'x': positions})],
    interactions=[PairCouplingInteraction('lj', 'atoms', 'atoms', cutoff=2.5, potential=lennard_jones)],
    space=ContinuousSpace(n_dim=2),
)
Simulation(liquid, dt=0.05).prepare().run(until=1).trajectory_.values('atoms.x').shape
```

A pair interaction is written once for a pair of members and summed over every pair closer than its cutoff, inside one
group or between two groups. `PairCouplingInteraction` adds to the rate of a state of both members a function of the
pair, or minus the gradient of a pair potential of their distance, divided by an optional `divisor`. On a state of
order 2, such as the position of a `PointElement`, the rate is its second derivative.

`period` gives the side of a periodic box, so pair distances use the nearest periodic image.

#### Boundaries

Boundaries keep members inside a box, by walls that reflect them or by edges that wrap around. The gas at the top of
this section bounces off the walls of its `ContinuousSpace(boundary='reflecting')`.

- A `BoundaryLaw` sets the value of a state after every internal solver step, however long the frame, so a frame never
  reports a state that a boundary would change.
- Adaptive solvers find the first time in a step where a boundary acts, and restart there.
- Boundaries combine with every form, so walls and periodic boxes work for differential equations, stochastic
  systems, jump processes, and discrete maps.
- A `ContinuousSpace` with `boundary='reflecting'` mirrors the position states of its elements at the walls, and
  reverses the first derivatives of the positions of order 2. With `boundary='periodic'`, it wraps positions around
  the box.

### Lattice systems

Use a lattice for systems that live on a grid of cells, such as cellular automata, sandpiles, and spin systems. You
write the laws for one cell, and the library runs them on every site.

```python
from skcomplex.core.components.spaces import LatticeSpace

space = LatticeSpace((4, 3)).prepare()
space.size_, space.shape_, space.periodic_, space.positions_[4].tolist()
```

- `LatticeSpace(shape, boundary=...)` is a box of `Z^d` with `'periodic'`, `'fixed'`, or `'reflecting'` edges.
- `LatticeElement(label, element=..., space=space, values={...})` puts one member of an element type on every site,
  with values given as arrays of the lattice shape.

A space is a part of its system and speaks the same verbs. `space.prepare()` checks its arguments and derives its
facts, whose names end in `_`:

- `size_` is the number of sites, and `n_dim_` the dimension.
- `shape_` is the shape of the arrays of its sites, and `positions_` the position of every site.
- `periodic_` says whether it wraps, and `default_visual_` is the visual of its groups.

Compiling a system prepares its spaces, so you call `prepare()` on a space only when you read its facts before that.

#### Local rules

Use a local rule when a cell's next value depends on the values of its neighbours. Conway's Game of Life is one, and
a glider moves one cell diagonally every four steps:

```python
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import LatticeSpace, MooreNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import OuterTotalisticSpec, StateSpec


class LifeElement(Element):
    def __init__(self, label, *, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive

    def _get_laws(self, s):
        return [LocalRuleLaw(s.alive, MooreNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))]


glider = np.zeros((20, 20))
glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
space = LatticeSpace((20, 20))
cells = System(elements=[LatticeElement('cells', element=LifeElement, space=space, values={'alive': glider})])
trajectory = Simulation(cells, dt=1).prepare().run(until=4).trajectory_
np.array_equal(trajectory.lattice('cells.alive')[-1], np.roll(glider, (1, 1), axis=(0, 1)))
```

`LocalRuleLaw(targets, over, rule)` computes a cell's next value from the ordered values of its neighbourhood.

- `MooreNeighborhood`, `VonNeumannNeighborhood`, `BoxNeighborhood`, or `LatticeNeighborhood(offsets)` give the
  offsets of the neighbours.
- A rule is a `TableSpec` of patterns, a `TotalisticSpec` or `OuterTotalisticSpec` rule, a rule code whose `table()`
  returns a table, or a function of NumPy arrays, one per neighbour. A function also receives the parameters it names
  and, for stochastic rules, uniform random numbers.
- Tables and totalistic rules run as lookups, fused into one compiled loop on the Numba backend.
- Rules update synchronously by default, every cell from the previous configuration. `schedule='random-sequential'`,
  `'random-site'`, `'checkerboard'`, or `'colored'` update cells one at a time, or one class of sites at a time.
- One unit of time is one application of every rule, so `dt` is an integer. `trajectory.lattice(name)` returns the
  states in lattice shape.
- `DriveLaw(target, function)` changes a whole lattice at once, such as dropping a grain on a random site.

#### Cascades

Use a cascade when a slow drive builds up stress and a fast relaxation releases it in avalanches, as in
self-organised criticality. The sandpile of the catalog records the size of every avalanche:

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

sandpile = create_system('sandpile', shape=(20, 20))
trajectory = Simulation(sandpile, dt=100, random_state=0).prepare().run(until=2000).trajectory_
max(event.size for event in trajectory.events)
```

`CascadeLaw(drive, relax, stable)` separates the slow drive from the fast cascade. Every step applies the drive laws
once, and the relax laws until `stable` holds. Each cascade is recorded among the trajectory's events with its size,
duration, and area. `activity` marks the sites that count as updates, such as the toppling sites of a sandpile.

#### Monte Carlo

Use a sampling law to draw the configurations of a lattice at a temperature, such as an Ising magnet, whose spins
align below the critical temperature:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import LatticeSpace, VonNeumannNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import MoveSpec, ObjectiveSpec, ParameterSpec, SamplerSpec, StateSpec

nearest = VonNeumannNeighborhood()


class SpinElement(Element):
    def __init__(
        self,
        label,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec()] = 1.5,
    ):
        super().__init__(label)
        self.spin = spin
        self.J = J
        self.temperature = temperature

    def _get_laws(self, s):
        energy = -s.J * s.spin * sum(neighbors(s.spin, nearest, reduce='each'))
        sampler = SamplerSpec('metropolis', 1 / s.temperature)
        return [SampleLaw(ObjectiveSpec(energy, nearest), MoveSpec(s.spin, 'flip'), sampler)]


spins = System(elements=[LatticeElement('spins', element=SpinElement, space=LatticeSpace((32, 32)))])
abs(Simulation(spins, dt=50, random_state=0).prepare().run(until=200).state_.mean()) > 0.9
```

A `SampleLaw(objective, move, sampler)` samples states.

- `ObjectiveSpec(expression, over)` is the local objective of a site, every term of the total objective that involves
  it. Neighbour values are read through `neighbors`.
- `MoveSpec(target, kind)` proposes changes, with the kinds `'flip'`, `'random-value'`, and `'rotate'`.
- `SamplerSpec('metropolis' | 'glauber' | 'heat-bath', intensity)` accepts them.

The change of the objective is derived symbolically, and sweeps update the two sublattices in turn. The intensity is
an expression of parameters, so it can change while the simulation runs.

#### Triangular lattices

Use a triangular lattice when every site needs six neighbours, as hexagonal cells and lattice gases do. A fire spreads
from one site to its six neighbours at every step:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import LatticeSpace, TriangularNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


def spread(*values):
    return np.max(values, axis=0)


class BurnElement(Element):
    def __init__(self, label, *, fire: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.fire = fire

    def _get_laws(self, s):
        return [LocalRuleLaw(s.fire, TriangularNeighborhood(center=True), spread)]


fire = np.zeros((40, 40))
fire[20, 20] = 1
space = LatticeSpace((40, 40), geometry='triangular')
forest = System(elements=[LatticeElement('forest', element=BurnElement, space=space, values={'fire': fire})])
burning = Simulation(forest, dt=1).prepare().run(until=3).trajectory_.lattice('forest.fire')
[int(item.sum()) for item in burning]
```

- `LatticeSpace(shape, geometry='triangular')` keeps integer coordinates on a skewed grid, so every boundary works as
  on a square lattice.
- It places site `(i, j)` at `(j + i / 2, i sqrt(3) / 2)` in the plane, given by the fitted `positions_`.
- `TriangularNeighborhood()` holds the six neighbours counterclockwise from 0 degrees, so direction `k + 3` modulo 6 is
  opposite to direction `k`, as lattice gases need.
- The cells of the sites are the hexagons of a hexagonal tiling.

#### Lattices on graphs

Use a graph as the lattice when the sites are the nodes of a network, such as an Ising magnet on a random regular
graph:

```python
from typing import Annotated

import networkx as nx
from sage.all__sagemath_symbolics import ZZ

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import SampleLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import AdjacencyNeighborhood, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import MoveSpec, ObjectiveSpec, ParameterSpec, SamplerSpec, StateSpec


class GraphSpinElement(Element):
    def __init__(
        self,
        label,
        *,
        spin: Annotated[float, StateSpec(domain=ZZ)] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        temperature: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.spin = spin
        self.J = J
        self.temperature = temperature

    def _get_laws(self, s):
        energy = -s.J * s.spin * neighbors(s.other.spin, reduce='sum')
        sampler = SamplerSpec('metropolis', 1 / s.temperature, schedule='colored')
        return [SampleLaw(ObjectiveSpec(energy, AdjacencyNeighborhood()), MoveSpec(s.spin, 'flip'), sampler)]


graph = nx.random_regular_graph(4, 200, seed=0)
spins = System(elements=[LatticeElement('spins', element=GraphSpinElement, space=NetworkSpace(graph))])
abs(Simulation(spins, dt=50, random_state=0).prepare().run(until=200).state_.mean()) > 0.9
```

`LatticeElement` also takes a `NetworkSpace`, with one site per node in the order of `graph.nodes`. Initial values are
arrays in node order, or mappings from node to value.

- `AdjacencyNeighborhood(center=False)` is the neighbourhood of a node, its neighbours along its edges, padded to the
  largest degree with the network's `fill` value.
- Local rules, drives, relaxations, Monte Carlo laws, and extremal laws run on graphs as on lattices.
- Lookup rules on graphs take the totalistic forms. A function rule that names the keyword `degree` receives the
  number of real neighbours of every site.
- The padded table grows with the largest degree. Graphs with hubs of very high degree, beyond scale-free graphs of
  about 10^5 nodes, therefore use much memory.

#### Neighbour sums and colored schedules

The graph example above uses two tools that work on every lattice.

- `neighbors(expression, reduce='sum')` sums an expression of a site and one neighbour, read through `s.other`, over
  the real neighbours of the site. Local energies are therefore exact on sites of any degree and outside fixed edges.
- `schedule='colored'` on local rules and samplers updates the classes of a proper coloring of the neighbour graph in
  turn. NetworkX computes the coloring once per law. The checkerboard schedule stays the two-color case of square
  lattices.

#### Extremal updates

Use an extremal update when each step acts on the worst or best site, such as the least fit species of the Bak-Sneppen
system of evolution, which is replaced together with its neighbours:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import ExtremalLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


def renew(fitness, random):
    return random


class SpeciesElement(Element):
    def __init__(self, label, *, fitness: Annotated[float, StateSpec()] = 0.5):
        super().__init__(label)
        self.fitness = fitness

    def _get_laws(self, s):
        return [ExtremalLaw(s.fitness, LatticeNeighborhood(((-1,), (1,))), renew, repeat=100, label='extinction')]


fitness = np.random.default_rng(0).random(100)
space = LatticeSpace(100)
ring = System(elements=[LatticeElement('ring', element=SpeciesElement, space=space, values={'fitness': fitness})])
trajectory = Simulation(ring, dt=10, random_state=0).prepare().run(until=100).trajectory_
len(trajectory.events), trajectory.events[0].label
```

`ExtremalLaw(targets, over, rule, key=None, extremum='min', repeat=1)` finds the site with the smallest or largest
value of a key state. It breaks ties with the simulation's random stream, and applies
`rule(*values, random, **parameters)` to the site and its real neighbours, `repeat` times per step.

- Every application is recorded among the trajectory's events as an `ExtremalRecord` with its time, site, and
  extremal value.
- On the Numba backend, the applications of a step run in one compiled loop when Numba compiles the rule. The rule
  then takes its arguments by position or keyword, with no keyword-only arguments.
- Both backends give the same events.

#### Viewing lattices and ready-made lattice systems

Square lattice groups stream to Rerun as images colored by value. Triangular lattices and graph lattices stream as
sites with `SiteVisual`: hexagonal cells at the site positions, or nodes colored by state with their edges logged once.

The catalog holds ready-made lattice systems, each under the field path of its science, such as
`computer_science/automata` for `game_of_life` and `physics/statistical_mechanics` for `ising`:

```python
from skcomplex.systems import catalog

lattice_systems = [
    'elementary_automaton', 'game_of_life', 'majority_rule', 'langtons_ant', 'nagel_schreckenberg', 'hodgepodge',
    'sandpile', 'forest_fire', 'hexagonal_forest_fire', 'earthquakes', 'ising', 'graph_ising', 'potts', 'xy',
    'fhp_lattice_gas', 'bak_sneppen',
]
fields = {entry.name: entry.field for entry in catalog()}
{name: fields[name] for name in lattice_systems}
```

### Networks

Put your elements on a graph when who meets whom matters more than where they stand. A `NetworkSpace` holds a NetworkX
graph. Its nodes carry the members of groups on its sites, and its edges carry the members of groups on its edges.

```python
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace

space = NetworkSpace(nx.karate_club_graph()).prepare()
space.edges_[None][0][:5].tolist(), space.edges_[None][1][:5].tolist()
```

The states and the structure of a network can change together. Interactions read and change the states of edges, laws
reduce over the neighbours of a node, and topology requests add and remove nodes and edges.

#### Generators and layers

Build a network with any NetworkX generator, and add a second relation over the same nodes as a layer.

```python
import networkx as nx

from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.specs import LayerSpec

first = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
second = NetworkSpace(nx.watts_strogatz_graph(100, 6, 0.1, seed=7))
sorted(first.graph.edges) == sorted(second.graph.edges)

space = NetworkSpace(nx.path_graph(4), layers=[LayerSpec('work', nx.complete_graph(4))]).prepare()
list(space.graphs_), [item.tolist() for item in space.edges_['work']]
```

- A `NetworkSpace` takes any NetworkX graph, so the generators of NetworkX build it, such as `nx.erdos_renyi_graph`,
  `nx.watts_strogatz_graph`, or `nx.barabasi_albert_graph`.
- A generator takes a seed, so a network drawn for an experiment is reproducible.
- Node positions in the node attribute `pos`, as `nx.random_geometric_graph` sets them, become the fitted `positions_`.
- A `LayerSpec` names a second relation over the same nodes, such as work next to contacts.

Preparing a network lowers every layer to arrays, keyed by its label, with `None` for the graph itself. `graphs_` holds
the graph of each layer, `edges_` the source and target of every edge, and `adjacency_` the neighbours of every node in
compressed sparse row form.

#### Interactions over edges

Give every edge a state of its own, such as the weight of a synapse, and change it from the states at both ends.
Hebbian weights grow with the product of the activities at the ends of each edge:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import NeighborInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class NeuronElement(Element):
    def __init__(self, label, *, a: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.a = a


class SynapseElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 0.1):
        super().__init__(label)
        self.w = w


class HebbianInteraction(NeighborInteraction):
    def __init__(
        self,
        label,
        *,
        neurons: Annotated[str, RoleSpec()],
        synapses: Annotated[str, RoleSpec()],
        eta: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.neurons = neurons
        self.synapses = synapses
        self.eta = eta

    def _get_laws(self, s):
        return [RateLaw(s.edge.w, s.eta * s.source.a * s.target.a)]


space = NetworkSpace(nx.cycle_graph(6))
system = System(
    elements=[
        LatticeElement('neurons', element=NeuronElement, space=space, values={'a': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]}),
        GroupElement('synapses', element=SynapseElement, space=space, on='edges'),
    ],
    interactions=[HebbianInteraction('hebb', neurons='neurons', synapses='synapses')],
)
trajectory = Simulation(system, dt=0.5).prepare().run(until=1.0).trajectory_
np.round(trajectory.values('synapses.w')[-1], 3).tolist()
```

- A group on the edges of a network, `GroupElement(..., on='edges')`, holds one member per edge of its layer.
- A `NeighborInteraction` binds a group on the nodes and, optionally, a group on the edges of the same network.
- Its laws run once per edge. They read `s.source` and `s.target`, the members at both ends, and `s.edge`, the member
  of the edge.
- Rate, noise, and jump laws may target any of the three, and update laws the edge member.
- The argument `layer` chooses the layer of an interaction without an edge group.

#### Hypergraphs and laws per hyperedge

Couple more than two nodes at once, such as three oscillators that pull on each other only as a triad. Three-body phase
coupling over the triangles of a hypergraph:

```python
import itertools
from typing import Annotated

import networkx as nx
import numpy as np
import xgi
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import HyperedgeInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class PhaseElement(Element):
    def __init__(self, label, *, theta: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.theta = theta


class TriadInteraction(HyperedgeInteraction):
    def __init__(self, label, *, phases: Annotated[str, RoleSpec()], K: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, size=3)
        self.phases = phases
        self.K = K

    def _get_laws(self, s):
        first, second = (other.theta for other in s.others)
        return [RateLaw(s.member.theta, s.K * sin(first + second - 2 * s.member.theta))]


triangles = xgi.Hypergraph([list(nodes) for nodes in itertools.combinations(range(5), 3)])
space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles)
system = System(
    elements=[LatticeElement('phases', element=PhaseElement, space=space, values={'theta': np.linspace(0, 1, 5)})],
    interactions=[TriadInteraction('triads', phases='phases')],
)
problem = compile_system(system)
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
len(space.prepare().hyperedges_), np.round(drift, 3).tolist()
```

A network space holds hyperedges beside its edges. `NetworkSpace(graph, hyperedges=...)` takes one of:

- sequences of at least two nodes,
- an XGI `Hypergraph`,
- an XGI `SimplicialComplex`, whose faces of at least two nodes are its hyperedges.

The members of an XGI hyperedge follow the order of the nodes of the graph. The prepared space holds `hyperedges_`, the
node indices of every hyperedge, `incidence_`, the sparse incidence matrix of nodes and hyperedges, and `memberships_`,
the hyperedges around every node.

A `HyperedgeInteraction` binds a group on the nodes and runs its laws once per hyperedge. It runs on the hyperedges of
its `size`, or of every size when the size is `None`. The laws read:

- `s.member`, the focal member,
- `s.others`, the other members in hyperedge order,
- `s.members`, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with `over=s.members`.

A law of the focal member runs once per member of every hyperedge and adds into that member. The compiler builds one
gather table per size of hyperedge, so the laws run as gathers and scatter-adds on the NumPy and Numba backends.

A second role may bind a group on the hyperedges, read as `s.hyperedge`, or a group on the edges, read as `s.edge`, the
edge that joins the first two members of the hyperedge. A law that targets either runs once per hyperedge.

The compiler reports two mistakes with the issue code `binding`: an interaction whose size no hyperedge of the network
has, and an edge group whose edge does not join the first two members of a hyperedge.

##### Symbolic groupings

Let members meet in small groups that form anew at every step, such as voters who talk in random groups of three. The
Galam majority rule on random groups of three, and the public goods of the groups of a lattice summed into a payoff:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import GroupingInteraction
from skcomplex.core.components.laws.utils import mean, where
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class PlayerElement(Element):
    def __init__(
        self,
        label,
        *,
        s: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.s = s
        self.payoff = payoff


opinions = (np.random.default_rng(0).random(30) < 0.7).astype(float)
voters = GroupElement('voters', element=PlayerElement, size=30, values={'s': opinions})
galam = GroupingInteraction(
    'galam',
    target='voters',
    groups='partition',
    size=3,
    form='symbolic',
    write='s',
    function=lambda g: where(mean(g.members.s, over=g.members) > 1 / 2, 1, 0),
)
run = Simulation(System(elements=[voters], interactions=[galam]), random_state=0).prepare().run(until=5)
run.trajectory_.values('voters.s').mean(axis=1).round(2).tolist()

players = LatticeElement(
    'players',
    element=PlayerElement,
    space=LatticeSpace((10, 10), neighborhood='von_neumann'),
    values={'s': (np.random.default_rng(1).random(100) < 0.5).astype(float)},
)
goods = GroupingInteraction(
    'goods',
    target='players',
    groups='neighborhood',
    form='symbolic',
    write='payoff',
    function=lambda g: 4.5 * mean(g.members.s, over=g.members) - g.member.s,
)
run = Simulation(System(elements=[players], interactions=[goods])).prepare().run(until=1)
run.trajectory_.values('players.payoff')[-1][:5].round(2).tolist()
```

`GroupingInteraction(form='symbolic', write=..., function=..., reduce='sum')` runs a function of one group on the same
path as the laws per hyperedge. The groups of a step are its hyperedges.

- The function reads the namespace of one group: `s.member`, the member it gives a value to, `s.others`, and
  `s.members`.
- It returns the value of the state `write` for that member in the next step, so the system runs in discrete time.
- A member takes the sum of the values of the groups that hold it, or the mean with `reduce='mean'`. A member in no
  group keeps its value.

The groups come in four kinds:

- `'pairs'` and `'partition'` are drawn again at every step from the random stream of the simulation. Two runs with one
  seed draw the same groups, on the NumPy and the Numba backends alike.
- `'neighborhood'` takes every member and the members around it in the space, fixed for the run.
- `'state'` groups the members that share a value of the state `by`, and every reduction over `s.members` reads only
  them.

`act` and `capacity` belong to the agent form, and `prepare` rejects them in the symbolic form.

##### Coupling over the hyperedges of a member

Couple every member to the hyperedges around it with one ready-made interaction, without writing an interaction
class. The three-body Kuramoto coupling over the triangles of a hypergraph, and the payoff of the public goods game
summed over the hyperedges of every size around a player:

```python
import itertools
from typing import Annotated

import networkx as nx
import numpy as np
from sage.all__sagemath_symbolics import sin

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.laws.utils import count, total
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class PlayerElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec()] = 0.0,
        s: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.s = s
        self.payoff = payoff


triangles = [list(nodes) for nodes in itertools.combinations(range(5), 3)]
space = NetworkSpace(nx.complete_graph(5), hyperedges=triangles)
triads = CouplingInteraction(
    'triads',
    target='oscillators',
    mode='hyperedges',
    size=3,
    read='theta',
    write='theta',
    function=lambda s: sin(s.others[0].theta + s.others[1].theta - 2 * s.member.theta),
)
oscillators = LatticeElement('oscillators', element=PlayerElement, space=space, values={'theta': np.linspace(0, 1, 5)})
problem = compile_system(System(elements=[oscillators], interactions=[triads]))
drift = problem.states.read(problem.drift(0.0, problem.initial_state, problem.initial_parameters), 'oscillators.theta')
np.round(drift, 3).tolist()

groups = NetworkSpace(nx.complete_graph(5), hyperedges=[[0, 1, 2], [1, 2, 3, 4], [0, 4]])
players = LatticeElement('players', element=PlayerElement, space=groups, values={'s': [1.0, 0.0, 1.0, 1.0, 0.0]})
goods = CouplingInteraction(
    'goods',
    target='players',
    mode='hyperedges',
    form='map',
    write='payoff',
    function=lambda s: 3 * total(s.members.s, over=s.members) / count(over=s.members) - s.member.s,
)
run = Simulation(System(elements=[players], interactions=[goods])).prepare().run(until=1)
run.trajectory_.values('players.payoff')[-1].round(2).tolist()
```

The reduction `hyperedges(expression, reduce='sum', size=None, weight=None)` of `skcomplex.core.components.laws.utils`
does the work. It reduces an expression written for one hyperedge over the hyperedges around every member of a group on
the nodes of a network with hyperedges, as `neighbors` does for one neighbour. Inside the expression:

- `s.others` holds the other members of the hyperedge in its order,
- `s.members` holds all its members, which `total`, `mean`, `count`, `maximum`, and `minimum` reduce with
  `over=s.members`,
- `s` is the reading member.

The reductions are those of `neighbors`. Without a size the expression reads `s.members`, and only the sum, the mean,
and the count run over hyperedges of several sizes. A law that reads it outside a network with hyperedges gets the
issue code `law-space`, and a size the network lacks gets the code `binding`.

`CouplingInteraction(mode='hyperedges', size=...)` couples every member through this reduction. The function reads
`s.member`, `s.others`, cut to `size - 1` members when the size is given, and `s.members`. The default term is the sum
over the other members of the difference of `read`. In the map form it sets the next value of `write` in discrete time.

##### Ordered phases of update laws

Let a later rule of a step read what an earlier rule of the same step wrote, such as an imitation that reads the
payoff just played. The public goods game of a lattice plays in the phase `play` and copies a random neighbour by the
Fermi rule in the phase `imitate`:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import GroupingInteraction, ImitationInteraction
from skcomplex.core.components.laws.utils import mean
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import PhaseSpec, PhasesScheduleSpec, StateSpec


class PlayerElement(Element):
    def __init__(
        self,
        label,
        *,
        s: Annotated[float, StateSpec()] = 0.0,
        payoff: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.s = s
        self.payoff = payoff


players = LatticeElement(
    'players',
    element=PlayerElement,
    space=LatticeSpace((20, 20), neighborhood='von_neumann'),
    values={'s': (np.random.default_rng(0).random(400) < 0.5).astype(float)},
)
play = GroupingInteraction(
    'play',
    target='players',
    groups='neighborhood',
    form='symbolic',
    write='payoff',
    function=lambda g: 5.5 * mean(g.members.s, over=g.members) - g.member.s,
    phase='play',
)
imitation = ImitationInteraction(
    'imitation', target='players', state='s', rule='fermi', form='probability', intensity=2.0, phase='imitate'
)
time = DiscreteTime(schedule=PhasesScheduleSpec((PhaseSpec('play'), PhaseSpec('imitate'))))
system = System(elements=[players], interactions=[play, imitation], time=time)
run = Simulation(system, random_state=0).prepare().run(until=30)
round(float(run.trajectory_.values('players.s')[-1].mean()), 2) > 0.5
```

By default every update law of a step reads the states at the start of the step, so a payoff and an imitation that
reads it would need two steps. Phases fix this.

- `UpdateLaw(state, value, phase=...)` names a phase. The interactions with update laws pass a `phase` to theirs:
  `GroupingInteraction` in its symbolic form, `ImitationInteraction` in its probability form, and `CouplingInteraction`
  in its map form.
- The phases of the `PhasesScheduleSpec` of the `DiscreteTime` run one after the other within a step, on the NumPy and
  Numba backends. Each reads the states the phases before it set.
- Laws without a phase run in the default phase, first unless the schedule places it.
- The draws of a step are made once, before the first phase.
- A law in a phase the schedule does not declare raises `ValidationError`.

#### Reductions over neighbourhoods

Sum, average, or count a quantity over the neighbours of every member, and weight it by the state of the edge between
them. Diffusion on a path whose second edge conducts ten times better:

```python
from typing import Annotated

import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement, LatticeElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.specs import RoleSpec, StateSpec


class NodeElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u


class LinkElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.w = w


class DiffusionInteraction(Interaction):
    def __init__(self, label, *, nodes: Annotated[str, RoleSpec()], links: Annotated[str, RoleSpec()]):
        super().__init__(label)
        self.nodes = nodes
        self.links = links

    def _get_laws(self, s):
        return [RateLaw(s.nodes.u, neighbors(s.nodes.other.u - s.nodes.u, weight=s.links.w))]


space = NetworkSpace(nx.path_graph(3))
system = System(
    elements=[
        LatticeElement('nodes', element=NodeElement, space=space, values={'u': [0.0, 1.0, 4.0]}),
        GroupElement('links', element=LinkElement, space=space, on='edges', values={'w': [1.0, 10.0]}),
    ],
    interactions=[DiffusionInteraction('diffusion', nodes='nodes', links='links')],
)
problem = compile_system(system)
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
problem.states.read(drift, 'nodes.u').tolist()
```

`neighbors(expression, over=..., reduce=..., weight=...)` reduces an expression written for one neighbour, through
`s.other`, over the neighbours of every member. The neighbours are:

- the neighbours of a network, by default,
- a layer, with `AdjacencyNeighborhood(layer=label)`,
- the neighbourhood of a lattice,
- in a continuous space, `RadiusNeighborhood(r)`, `NearestNeighborhood(k)`, and `ConeNeighborhood(r, angle)`.

The reductions are `'sum'`, `'mean'`, `'count'`, `'max'`, `'min'`, and `'normalized-mean'`. A weight may read the state
of the edge group of the layer through a role of an interaction, as the weight of the edge that joins the member to its
neighbour. The Kuramoto coupling on a network is
`RateLaw(s.theta, s.omega + s.K * neighbors(sin(s.other.theta - s.theta), reduce='mean'))`.

Validation reports a reduction whose neighbourhood the space does not offer, such as `RadiusNeighborhood(1.0)` on a
network, with the code `law-space`, and a layer the network lacks with the code `binding`.

##### Connected components of a network

Find the clusters of connected members, for percolation or for the giant component of an epidemic. The size of the
cluster of occupied sites around every site of a path:

```python
from typing import Annotated

import networkx as nx

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import components, where
from skcomplex.core.components.spaces import NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.specs import StateSpec


class SiteElement(Element):
    def __init__(self, label, *, occupied: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.occupied = occupied

    def _get_laws(self, s):
        both = where(s.occupied > 0, 1, 0) * where(s.other.occupied > 0, 1, 0)
        return [RateLaw(s.occupied, components(both, reduce='size'))]


sites = LatticeElement(
    'sites',
    element=SiteElement,
    space=NetworkSpace(nx.path_graph(6)),
    values={'occupied': [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]},
)
problem = compile_system(System(elements=[sites]))
drift = problem.drift(0.0, problem.initial_state, problem.initial_parameters)
problem.states.read(drift, 'sites.occupied').tolist()
```

`components(active, over=None, reduce='label')` reduces over the connected components of the active links between the
members of a group. The links are the pairs of neighbours of the neighbourhood, the graph of the network by default.
The expression `active` is written for one link, as in `neighbors`. It reads `s.other`, `s`, and the edge group of a
role, so a condition on both ends gives site percolation and an edge weight gives bond percolation. The reductions are:

- `'label'`, the smallest member of the component of every member,
- `'size'`, the size of that component,
- `'largest'`, the size of the largest component,
- `'giant'`, 1 for the members of the largest component.

The NumPy backend runs SciPy's `connected_components` on the active links, and the Numba backend a union-find.

#### Topology requests

Grow, shrink, and rewire a network while it runs. Barabási-Albert growth adds one node with two edges at every step,
and the new edges prefer nodes of high degree:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec, TopologyRequestSpec


class NodeElement(Element):
    def __init__(self, label, *, age: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.age = age

    def _get_laws(self, s):
        return [UpdateLaw(s.age, s.age + 1)]


class LinkElement(Element):
    pass


def degree(node):
    return neighbors(1)


class GrowthElement(Element):
    def _get_laws(self, s):
        return [UpdateLaw(TopologyRequestSpec('add-node', edges=2, toward=degree), 1)]


system = System(
    elements=[
        GroupElement('nodes', element=NodeElement, on='sites', variable=True, capacity=203),
        GroupElement('links', element=LinkElement, on='edges', variable=True, capacity=403),
        GrowthElement('growth'),
    ],
    space=NetworkSpace(nx.complete_graph(3)),
    time=DiscreteTime(),
)
trajectory = Simulation(system, dt=50.0, random_state=1).prepare().run(until=200).trajectory_
snapshot = trajectory.network(-1)
degrees = np.bincount(np.concatenate([snapshot.sources, snapshot.targets]))
snapshot.nodes.size, snapshot.sources.size, int(degrees.max()) > 10
```

A network that changes needs room to grow. A variable group on the sites or edges of a network,
`GroupElement(..., on='sites', variable=True, capacity=n)`, holds rows for up to `capacity` members, and marks the
members present as active. The NumPy and Numba backends run such groups with fixed array shapes:

- laws leave the free rows unchanged,
- reductions read the members present,
- neighbourhoods and interactions over neighbours read the edges present.

A change of structure is a `TopologyRequestSpec` of the kind `'add-node'`, `'remove-node'`, `'add-edge'`,
`'remove-edge'`, or `'rewire'`. It is the target of update and jump laws:

- `UpdateLaw(request, probability)` fires the request with a probability at every step, for every member or edge its
  law runs for.
- `JumpLaw(rate, {request: 1})` fires it at a rate.

The iteration and Gillespie steppers apply the requests between steps. A request that joins nodes draws the new ends
with weights from `toward`. This is a module-level function of the namespace of a candidate node, such as one that
returns `neighbors(1)` for its degree. A request on a group that is not variable gets the validation issue
`group-fixed`, and a full capacity raises `CapacityError`.

Trajectories hold the values of absent members as NaN, and `trajectory.network(frame)` returns the nodes and edges
present at a frame. Rerun draws the nodes present, and the edges at every frame where they change.

New nodes can also attach by their distance to the others. Here every new node draws an angle and links to the nodes
whose angle lies within 0.1 of its own:

```python
from typing import Annotated

import networkx as nx
import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.laws.utils import where
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class PlacedElement(Element):
    def __init__(self, label, visual=None, *, angle: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label, visual=visual)
        self.angle = angle


angles = np.linspace(0.0, 6.0, 80).tolist()
nodes = GroupElement('nodes', element=PlacedElement, on='sites', variable=True, capacity=130, values={'angle': angles})
links = GroupElement('links', element=Element, on='edges', variable=True, capacity=130)
growth = TopologyInteraction(
    'growth',
    action='add-node',
    values={'angle': lambda rng: rng.uniform(0.0, 6.0)},
    toward=lambda node: where(abs(node.angle - node.source.angle) <= 0.1, 1, 0),
)
space = NetworkSpace(nx.empty_graph(80))
system = System(elements=[nodes, links], interactions=[growth], space=space, time=DiscreteTime())
trajectory = Simulation(system, random_state=0).prepare().run(until=50).trajectory_
network = trajectory.network(-1)
placed = trajectory.values('nodes.angle')[-1]
network.sources.size, bool((abs(placed[network.sources] - placed[network.targets]) <= 0.1).all())
```

- The weight `toward` reads the node a new edge starts from as `node.source`, beside the candidate. That node is the new
  node of `'add-node'`, with the values it drew, the adding node of `'add-edge'`, and the kept end of `'rewire'`.
- Without a target, the condition and a probability of `'add-node'` read the drawn values of the node being added as
  `s.source`, so a node that does not fire is never added. A rate does not read them, since no node exists before the
  jump, and neither does a reduction.
- With a target, the condition keeps reading the node it runs for.
- Where fewer candidates have a positive weight than the edges to draw, the new ends are drawn uniformly.

The catalog systems `popularity_similarity_network` and `urban_scaling_interactions` grow by the hyperbolic distance.
The second adds internal links at a fixed rate per resident, and the exponent of its meetings rises with that rate. It
is near 1 without internal links, and near the 1.15 that Ribeiro and Netto (2024) review at the rate 0.002 over sizes
from 100 to 800.

A network can also grow by hyperedges. A network grows by triangles from one triangle:

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TopologyInteraction
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation

system = System(
    elements=[
        GroupElement('nodes', element=Element, on='sites', variable=True, capacity=23),
        GroupElement('links', element=Element, on='edges', variable=True, capacity=43),
        GroupElement('triangles', element=Element, on='hyperedges', variable=True, capacity=21),
    ],
    interactions=[TopologyInteraction('growth', action='add-node', hyperedge=3)],
    space=NetworkSpace(nx.complete_graph(3), hyperedges=[[0, 1, 2]]),
    time=DiscreteTime(),
)
trajectory = Simulation(system, random_state=0).prepare().run(until=20).trajectory_
trajectory.network(-1).hyperedges.shape, trajectory.network(-1).sources.size
```

`TopologyInteraction(action='add-node', hyperedge=k)`, or `TopologyRequestSpec('add-node', hyperedge=k)`, makes every
new node form a hyperedge of `k` members with `k - 1` existing nodes, and join each of them by an edge.

- The first two members are the ends of an edge drawn uniformly, the others are drawn by `toward`, and the new node is
  the last member. So an edge joins the first two members of every hyperedge.
- The hyperedges are the rows of a variable group on the hyperedges, `GroupElement(on='hyperedges', variable=True,
  capacity=n)`. The initial hyperedges and the new ones have one size.
- Hyperedge interactions read the hyperedges present at every step, and `trajectory.network(frame).hyperedges` holds
  the members of those present at a frame.

### Fields

Describe a quantity spread over space, such as heat in a rod or a chemical in a dish, and write its law once, for one
site. Heat spreads from a hot spot in the middle of a rod with insulated ends:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class HeatElement(Element):
    def __init__(
        self,
        label,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        D: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.u = u
        self.D = D

    def _get_laws(self, s):
        return [RateLaw(s.u, s.D * laplacian(s.u))]


initial = np.zeros(400)
initial[200] = 20.0
space = LatticeSpace(400, spacing=0.05)
rod = FieldElement('rod', element=HeatElement, space=space, boundary={'u': 'zero-flux'}, values={'u': initial})
trajectory = Simulation(System(elements=[rod]), dt=0.1).prepare().run(until=0.5).trajectory_
heat = trajectory.values('rod.u')
round(float(heat[-1].max()), 3), heat.sum(axis=1)[[0, -1]].round(6).tolist()
```

A field holds one value per site of a lattice or per node of a network. `FieldElement(label, element=..., space=...,
boundary=..., values=...)` places a member of an element type on every site. The laws of the type are written for one
site, and they read the neighbouring sites through differential operators. The method of lines turns a field into
ordinary differential equations, so every ODE solver and backend runs it.

- `laplacian(u)` lowers to the `2d + 1`-point stencil divided by the squared spacing, and to the graph Laplacian on a
  `NetworkSpace`.
- `gradient(u, n_dim)` and `divergence(vector)` use central differences, and `advection(u, velocity)` upwind
  differences.
- Operators apply to any expression of the field's states and parameters, and they nest.
- `boundary={'u': 'periodic' | 'zero-flux' | ('fixed', value)}` sets the edges of each state. Zero flux mirrors the
  field and conserves its total, and fixed edges read the given value. The lattice's own `boundary` is the default.
- The equations keep the operators, `diff(u(t), t) == D*laplacian(u(t))`, while the backends evaluate the stencils.
  On Numba, the stencils of a lattice are fused into one compiled loop.
- `problem.stable_step` is the diffusive limit `h^2 / (2 d D)` of explicit steps. Fixed-step solvers whose `dt` exceeds
  it warn before the run, with the limit.

Agents on a grid of the same shape couple to a field in co-simulation. A rule reads the field at a cell with
`system.field('ground/pheromone.level', cell)` and adds to it with `system.deposit(name, cell, amount)`. The deposits
reach the field at the next synchronisation point. Lattice fields stream to Rerun as images.

The library has `heat_diffusion`, `fisher_kpp`, `gray_scott`, `brusselator`, `fitzhugh_nagumo`, `schnakenberg`, and
`pheromone_foraging`, which you select by name, such as `create_system('gray_scott')`.

#### Discretizations and stiff solvers

Run the same laws with another discretization, and take long steps through stiff equations. The Kuramoto-Sivashinsky
equation, spectral in space and stepped by `ETDRK4Solver` far beyond the explicit limit:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import divergence, laplacian
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ETDRK4Solver
from skcomplex.core.specs import StateSpec


class ChaosElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u

    def _get_laws(self, s):
        return [RateLaw(s.u, -divergence([s.u**2 / 2]) - laplacian(s.u) - laplacian(laplacian(s.u)))]


positions = np.arange(1, 129) * 32 * np.pi / 128
line = FieldElement(
    'line',
    element=ChaosElement,
    space=LatticeSpace(128, spacing=32 * np.pi / 128),
    values={'u': np.cos(positions / 16) * (1 + np.sin(positions / 16))},
    discretization='spectral',
)
state = Simulation(System(elements=[line]), dt=10, solver=ETDRK4Solver(dt=0.25)).prepare().run(until=100).state_
round(float(np.abs(state).max()), 1)
```

The argument `FieldElement(..., discretization=...)` chooses how space is discretized:

- `'spectral'` fields on periodic lattices multiply Fourier modes, and the two-thirds rule removes aliasing from every
  operator, so `laplacian(u)` is exact for smooth fields. `inverse_laplacian(u)` solves Poisson equations.
  `symmetry={'u': (None, 'odd')}` keeps a state odd or even along an axis, which represents walls on a doubled lattice.
  `LatticeSpace(spacing=(dx, dy))` sets a spacing per axis.
- `'finite-volume'` fields take `FluxLaw(state, flux)`s, `du/dt + divergence(flux) = 0`. They are updated with MUSCL
  reconstruction, minmod or van Leer limiters, and Rusanov or exact Godunov fluxes. `SSPRKSolver` steps them by the CFL
  condition and by the stability limit of their Laplacian rates at the current parameters, and conserves their totals.
- `'finite-element'` fields live on a `MeshSpace` of the points and cells of a scikit-fem mesh, such as
  `MeshSpace(grid.p, grid.t)` for `grid = skfem.MeshTri.init_lshaped().refined(3)`. Their stiffness and mass matrices
  come from scikit-fem, and their boundary values can be functions of the coordinates. They stream to Rerun as colored
  meshes.

Stiff equations need implicit or exponential steps. The compiler splits the rate of every field into a linear part with
constant coefficients, such as `D * laplacian(u)` or `-laplacian(laplacian(u))`, and a nonlinear remainder.

- `IMEXEulerSolver`, `IMEX2Solver`, and `CrankNicolsonSolver` treat the linear part implicitly. They use sparse solves
  for finite differences and finite elements, and solves per Fourier mode for spectral fields.
- `ETDRK4Solver` integrates the linear part exactly per mode, coupled states included, so stiff equations take steps
  far beyond the explicit limit.

The implicit methods of `ScipyODESolver` are the stiff path for finite-difference and finite-element fields, with the
adaptive error control that the fixed-step split steppers lack:

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import ScipyODESolver
from skcomplex.systems import create_system

stiff = create_system('fisher_kpp', length=400, D=1.0, spacing=0.05)
simulation = Simulation(stiff, dt=1.0, solver=ScipyODESolver(method='BDF')).prepare().run(until=5)
simulation.stepper_.jacobian_pattern_.nnz, simulation.state_.max().round(3)
```

The compiler derives the sparsity pattern of the Jacobian from the laws. The pattern covers member and shared reads,
the neighbour tables of stencils, the sparse matrices of finite elements, and the reconstruction stencils of finite
volumes. It holds dense blocks where couplings are known only at run time, such as pairs, spectral operators, and
callables.

- `ScipyODESolver(method='BDF')` and `ScipyODESolver(method='Radau')` receive the pattern. SciPy then estimates the
  Jacobian with one drift evaluation per group of independent columns, and factors it with sparse LU.
- `LSODA` receives the half bandwidths of a pattern narrower than a tenth of the size.
- The option `jacobian` selects the path: `'auto'` by default, `'exact'` for the sparse Jacobian of the laws
  differentiated by Sage, which laws without a symbolic form cannot give, and `'dense'` for the dense estimate.
- Explicit methods ignore it, and `RK45` warns when one call spans more than 1000 stable steps of the diffusion of a
  field.

On Fisher-KPP on 4000 points, whose largest eigenvalue is about 1600, run to `t = 10`, BDF with the dense estimate took
35 to 50 s. With the pattern it took about 0.2 s, with the exact Jacobian 0.3 s, Radau with the pattern 0.7 s, and
`CrankNicolsonSolver(dt=0.01)` 0.8 s. Spectral fields keep `ETDRK4Solver` and the IMEX steppers. These win there,
because the linear part is diagonal in Fourier space while its pattern is dense.

The system `decaying_turbulence` is two-dimensional incompressible flow in vorticity-streamfunction form, and
`rayleigh_benard` adds Boussinesq convection, with `Ra` and `Pr`. The library also has `kuramoto_sivashinsky`,
`complex_ginzburg_landau`, `swift_hohenberg`, `heat_l_shape`, `burgers`, and `shallow_water`.

#### Nonlocal operators

Read a field beyond the nearest sites: smooth it against a kernel, solve for a potential, or look ahead by a distance.
Each rate below changes a Gaussian bump on a line in its own way:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import exp, pi, sqrt

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import convolve, displacement, inverse_helmholtz, partial, shifted
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class SiteElement(Element):
    def __init__(
        self,
        label,
        *,
        u: Annotated[float, StateSpec()] = 0.0,
        k: Annotated[float, ParameterSpec()] = 2.0,
    ):
        super().__init__(label)
        self.u = u
        self.k = k


def rate_of(rate):
    centers = np.arange(128) * 0.1
    values = {'u': np.exp(-((centers - 6.4) ** 2))}
    line = FieldElement('line', element=SiteElement, space=LatticeSpace(128, spacing=0.1), values=values)
    interaction = RateInteraction('rate', target='line', state='u', rate=rate)
    simulation = Simulation(System(elements=[line], interactions=[interaction]), dt=0.05)
    return simulation.prepare().run(until=0.2).trajectory_.values('line.u')


def smoothing(s):
    kernel = exp(-displacement(0) ** 2) / sqrt(pi)
    return convolve(s.target.u, kernel) - s.target.u


def screened_drift(s):
    return partial(inverse_helmholtz(s.target.u, s.target.k), 0)


def look_ahead(s):
    return shifted(s.target.u, [0.25 + s.target.u]) - s.target.u


{rate.__name__: rate_of(rate)[-1].sum().round(3) for rate in (smoothing, screened_drift, look_ahead)}
```

Three operators of `skcomplex.core.components.laws.utils` read beyond the nearest sites, and a law writes them as a
reference writes them:

- `inverse_helmholtz(expression, screening, boundary=None)` is the solution of
  `(laplacian - screening**2) psi = expression`, and `inverse_laplacian(u)` is its case without screening. It runs on
  spectral fields, and on finite-difference lattices with periodic, zero-flux, or fixed edges, where the fast transforms
  of SciPy invert the stencil that `laplacian` uses. On networks and meshes it uses a sparse factorization. An operand
  that mixes states of different boundaries needs `boundary=`.
- `convolve(expression, kernel, radius=None)` sums an expression over the sites of a lattice against a kernel written
  in the symbols `displacement(axis)` and the parameters. The kernel is sampled when the problem is compiled. It wraps
  around periodic edges and is cut at fixed and zero-flux edges, where the operand is padded with the fixed value or
  its mirror image.
- `shifted(expression, offset, order=1)` reads an expression at a physical displacement, one component per axis, which
  may depend on the states. A displacement of whole sites reads the neighbour tables and stays compilable to Numba. Any
  other one interpolates with `order=1` or `order=3`, following the boundary of the field.

The stencil operators read a nonlocal result at the neighbouring sites, so `partial` and `gradient` apply to an
inverted streamfunction. Laws with these operators run on the NumPy backend, except whole-site reads. The sparsity
pattern of the Jacobian marks a convolution as coupling the sites within its radius, and an inversion as dense. So
implicit solvers stay correct but slow, and explicit or exponential solvers suit these laws best. An observable may use
the operators too, which streams a streamfunction as an image.

The misuses are issues of the compiler's report, each with the path of the law and a fix:

- `law-space` for a convolution or a shifted read on a network or a mesh, and for a spectral operator on a lattice
  without periodic edges,
- `kernel-state` for a kernel that reads a state,
- `screening` for a screening that reads a state or whose square can be negative,
- `operator-spaces` for an operand that reads fields of more than one space.

A field can also couple its sites through a function of both ends, without an operator. A `CouplingInteraction` on a
field sums the function over the neighbours of every site, and reads a fixed boundary value as `laplacian` does:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.interactions import CouplingInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class CellElement(Element):
    def __init__(self, label, *, u: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.u = u


cells = FieldElement('cells', element=CellElement, space=LatticeSpace(32), values={'u': np.eye(32)[16]})
exchange = CouplingInteraction(
    'exchange',
    target='cells',
    read='u',
    write='u',
    function=lambda s: 0.2 * (s.other.u - s.member.u),
)
system = System(elements=[cells], interactions=[exchange])
trajectory = Simulation(system, dt=0.5).prepare().run(until=2.0).trajectory_
trajectory.values('cells.u')[-1].sum().round(6)
```

A constant times `s.other.u - s.member.u` lowers to `laplacian`, so spectral and finite-element fields keep their own
operator.

`amari_neural_field` holds a bump with a convolution, `double_gyre` streams the streamfunction of a closed basin,
`gkt_traffic` brakes on the traffic read ahead, and `active_viscoelastic_gel` solves its force balance at every step.

### Phase-space fields

Evolve a density over position and over classes such as velocity or orientation, the way a kinetic equation does.
Particles stream through position at their velocity, forces push them through velocity, and collisions redistribute
them over velocity at a fixed position.

A phase space is a lattice whose axes are the product of position and the class axes. So `FieldElement` holds one value
per point of it, and its laws are written for one point, like those of any field.

No existing library fits. `py-pde` stops at three dimensions and has no class axes, `pylbm` and `lbmpy` hold a few
discrete velocities, Dedalus is GPL with no wheels, and Struphy is specific to plasmas. No maintained Python library
holds the fast spectral collision operator. So the numerics wrap `scipy.fft` and `scipy.ndimage`, and the core holds
the adapters and the assembly of the collision weights.

#### Phase spaces and coordinates

Build a phase space from a position lattice and a velocity axis, and let a gas stream freely through it:

```python
import numpy as np

from skcomplex.core.components.elements import DensityElement, FieldElement
from skcomplex.core.components.interactions import AdvectionInteraction
from skcomplex.core.components.laws.utils import coordinate
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.specs import AxisSpec

space = PhaseSpace(LatticeSpace((32,), spacing=2 * np.pi / 32), [AxisSpec('v', 64, -6.0, 6.0)]).prepare()
x, v = np.arange(32) * 2 * np.pi / 32, space.coordinates_['v']
start = np.outer(1.0 + 0.5 * np.sin(x), np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))
gas = FieldElement('gas', element=DensityElement, space=space, values={'f': start}, discretization='finite-volume')
stream = AdvectionInteraction('stream', target='gas', state='f', drift=lambda s: [coordinate('v'), 0])
system = System(elements=[gas], interactions=[stream])
simulation = Simulation(system, dt=0.1, solver=RK4Solver(dt=0.01)).prepare()
final = simulation.run(until=0.5).trajectory_.values('gas.f')[-1]
abs(final.sum() / start.sum() - 1.0) < 1e-12
```

The total mass is conserved to round-off.

- `PhaseSpace(position, axes, limit=2**26)` takes a position `LatticeSpace` and the class axes.
- Each class axis is an `AxisSpec(name, size, low, high, periodic=False)` with its own cells. An interval has no flux
  through its ends, and a periodic axis, such as an angle, joins them.
- The position axes come first, and they keep the boundary of the lattice.
- The compile report holds a `space-size` issue for a space above its limit, with the size it would take.

`coordinate(name)` is the coordinate, along a named axis, of the point a law updates. Streaming is the drift
`[coordinate('v'), 0]`, and acceleration by a force `F` is the drift `[0, F]`. The compile report holds a `law-space`
issue for a coordinate in a law whose group is not on a phase space with that axis.

Finite-volume fields, finite-difference fields with `riemann=None`, and spectral fields with `riemann=None` all stream
on a phase space. Spectral derivatives need the axis to be periodic.

#### Moments and mean fields

Read the density, mean velocity, or temperature at every position, and feed them back into the dynamics. A
Vlasov-Poisson drift accelerates the gas by the field of its own charge density:

```python
from skcomplex.core.components.laws.utils import coordinate, gradient, inverse_laplacian, moment


def poisson_drift(s):
    return [coordinate('v'), -gradient(inverse_laplacian(1 - moment(s.target.f, 0, axes=('v',))), 1)[0]]


poisson_drift.__name__
```

- `moment(expression, order, axes=names)` reduces over class axes at every position, with the cell widths as weights.
- It is a field over position, which laws read broadcast along the reduced axes. So `normalized=True` and
  `central=True` give the mean velocity and the temperature at every position.
- `reduce_axes(expression, names)` is the integral behind it, for the moments of `cos` and `sin` of an orientation.
- A drift can read a moment of its own field, and the operators of a field over position act along the position axes.

Fields over position and fields on a phase space read each other:

- A drift reads another field over the position lattice through the `sources` of `AdvectionInteraction` and
  `ConservationInteraction`, broadcast along the class axes.
- A `RateInteraction` over the position lattice reads a moment of a field on a phase space through its `sources`.

#### Encounters and equilibria

Move particles between classes at one position, and relax them towards a local equilibrium:

```python
from skcomplex.core.components.interactions import RelaxationInteraction
from skcomplex.core.components.laws.utils import maximum_entropy

relax = RelaxationInteraction('r', target='gas', state='f', goal=lambda s: maximum_entropy(s.target.f, axes=('v',)))
relax.state
```

`EncounterInteraction` on a phase space pairs classes at the same position. Without `field` it is a linear
redistribution: a candidate at class `h` moves to class `i` at the rate `eta[h]` with the probability `kernel[h, i]`. A
rate that is one expression may read the `coordinate` of the class of the candidate.

`maximum_entropy(expression, axes, order=2)` is the density `exp(a + b c + d |c|^2)` whose discrete moments equal those
of the expression at every position. Newton steps find it, so a relaxation towards it conserves the moments to
round-off.

#### Collisions

Add binary collisions between particles of different velocities, as in the Boltzmann equation.
`EncounterInteraction(label, candidate=..., state='f', cross_section=..., factor=None)` is the binary collision
integral over two interval class axes.

- The cross-section is a function of arrays of the relative speed and of the cosine of the deflection angle.
- It enters the fast spectral method of Mouhot and Pareschi through a singular value decomposition, a single term for
  Maxwell molecules.
- The method uses FFT products on a grid padded by `(3 + sqrt(2)) / 2`. It conserves the density exactly, and the
  momentum and the energy within its spectral accuracy.
- `factor` scales the collision frequency, such as a function of the density moment for the local Enskog equation.
- The compile report holds a `law-space` issue for a collision without two interval class axes.

#### Solvers

Step transport and collisions each with the method that suits it, and combine them:

```python
from skcomplex.core.engine.solvers import RK4Solver, SemiLagrangianSolver, StrangSplittingSolver

solver = StrangSplittingSolver(SemiLagrangianSolver(dt=0.1), RK4Solver(dt=0.01), dt=0.1)
solver.dt
```

- `SemiLagrangianSolver(dt)` advances fields whose fluxes are linear drifts `a u`, with `a` free of the state and
  constant along its own axis. It shifts each slice along its characteristic, in the splitting of Cheng and Knorr. It
  is stable at any Courant number and conserves the total on periodic axes. Other fluxes raise a named error.
- `StrangSplittingSolver(transport, collision, dt)` composes a half transport step, a full collision step, and a half
  transport step. The problem keeps the source rates, `problem.sources`, apart from the flux differences.
- `RK4Solver`, `ScipyODESolver`, and the other method-of-lines solvers accept phase-space fields as well.

#### Visualizing

`ImageVisual(slices=..., moments=...)` logs the plane that fixes every axis but two, such as `slices={'vy': 2}`. It
also logs the moments over the class axes, as a bar series over a one-dimensional position lattice and as an image over
a two-dimensional one.

`boltzmann_equation`, `bgk_equation`, `vlasov_equation`, and `enskog` in `physics/fluids`, and `bertin_droz_gregoire`
in `physics/active_matter`, are the kinetic equations of the library.

### Kinetic systems of active particles

Describe a population by how its members spread over an activity, such as wealth, an opinion, or the progression of a
tumour cell, and let encounters between members reshape that spread. This is the kinetic theory of active particles of
Bellomo et al. Encounters change the distribution, and its moments give the macroscopic quantities.

#### Distributions as site groups

Hold a distribution over nine wealth classes as a group with one member per class:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.specs import StateSpec


class WealthDensityElement(Element):
    def __init__(self, label, *, f: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0):
        super().__init__(label)
        self.f = f


classes = np.linspace(0.0, 1.0, 9)
profile = np.exp(-((classes - 0.5) ** 2) / 0.05)
wealth = GroupElement('wealth', element=WealthDensityElement, size=9, on='sites', values={'f': profile / profile.sum()})
wealth.values['f'].round(3).tolist()
```

A distribution over `n` activity values is a group on the sites of a lattice of `n` sites, one member per activity,
`GroupElement(label, element=E, size=n, on='sites')`. The element declares one real state, the density at the member's
activity. The activity values are an array that the laws and the moments receive. No distribution component is needed,
and a population of several functional subsystems is several such groups.

#### Encounters

Declare who meets whom and where they end up, and let the compiler derive the equations. Traders exchange wealth, and
cooperation and competition move them one class up or down:

```python continuation
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import EncounterLaw
from skcomplex.core.components.spaces import ContinuousTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec


def exchange_table(cooperation, competition):
    table = np.zeros((9, 9, 9))
    for candidate in range(9):
        for field in range(9):
            gap = int(np.sign(field - candidate))
            for outcome, probability in ((candidate + gap, cooperation), (candidate - gap, competition)):
                if gap and 0 <= outcome < 9:
                    table[candidate, field, outcome] += probability
            table[candidate, field, candidate] += 1 - table[candidate, field].sum()
    return table


class ExchangeInteraction(Interaction):
    def __init__(self, label, *, population: Annotated[str, RoleSpec()], eta: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.population = population
        self.eta = eta

    def _get_laws(self, s):
        return [EncounterLaw(s.population.f, s.population.f, s.eta, exchange_table(0.2, 0.3))]


economy = System(
    elements=[wealth],
    interactions=[ExchangeInteraction('exchange', population='wealth')],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
densities = Simulation(economy, dt=10.0).prepare().run(until=100).trajectory_.values('wealth.f')
densities.sum(axis=1)[[0, -1]].round(12).tolist(), densities[-1].round(3).tolist()
PassCompiler().prepare(economy).report_.ok
```

`EncounterLaw(candidate, field, rate, transition, test=None)` declares that a candidate particle at activity `h` meets a
field particle at activity `k` at the rate `eta[h, k]`, and ends at activity `i` with the probability `A[h, k, i]`. An
interaction returns it with roles bound to the candidate and field groups. The compiler derives the gain and loss
equations

`df_i/dt = sum_hk eta[h, k] A[h, k, i] f_h g_k - f_i sum_k eta[i, k] g_k`

without any derivation by hand. When `test` is given, the candidate moves into the `test` group, such as a cell that
mutates into another subsystem.

- `BilinearLaw(target, first, second, rate, outcomes=None)` changes the target at the signed rate `r[h, k]`. A positive
  rate creates a particle without removing the first density, at its activity or at an activity drawn from `outcomes`.
  A negative rate with the first density as the target removes it.
- Rates and probabilities are numbers or parameter expressions, scalars or arrays. Only the nonzero probabilities enter
  the lowering, as sparse contractions on NumPy and one fused loop on Numba.
- The two laws need continuous time, a lattice space, and real states. Validation reports a law whose system lacks
  them, with the fix to declare them.
- Before compilation, validation reports the encounters whose probabilities do not sum to one over the outcomes, with
  the candidate and field activities and the sum.
- A system with encounters has no exact Jacobian, so implicit solvers estimate it by finite differences.

#### External actions

Add a policy that acts on the population from outside. External actions need no primitive of their own.

- A micro-scale action, such as a welfare policy that acts on the transitions of the wealth classes, is an encounter
  whose field role binds a group with fixed states, a group no law changes.
- A macro-scale action, such as a recruitment of security forces proportional to the mean criminality, is a `RateLaw`
  that reads the moments of a distribution.

A welfare policy that lifts the poorest classes and taxes the richest:

```python continuation
class PolicyInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        population: Annotated[str, RoleSpec()],
        government: Annotated[str, RoleSpec()],
        tax: Annotated[float, ParameterSpec()] = 0.2,
    ):
        super().__init__(label)
        self.population = population
        self.government = government
        self.tax = tax

    def _get_laws(self, s):
        rates, table = np.zeros((9, 1)), np.zeros((9, 1, 9))
        for candidate in range(9):
            outcome = candidate + 1 if candidate < 3 else candidate - 1 if candidate >= 6 else candidate
            rates[candidate, 0] = float(outcome != candidate)
            table[candidate, 0, outcome] = 1.0
        return [EncounterLaw(s.population.f, s.government.f, rates * s.tax, table)]


welfare = System(
    elements=[wealth, GroupElement('government', element=WealthDensityElement, size=1, values={'f': 1.0})],
    interactions=[
        ExchangeInteraction('exchange', population='wealth'),
        PolicyInteraction('policy', population='wealth', government='government'),
    ],
    space=LatticeSpace(9, boundary='fixed'),
    time=ContinuousTime(),
)
Simulation(welfare, dt=10.0).prepare().run(until=100).trajectory_.values('wealth.f')[-1].round(3).tolist()
```

#### Moments

Record the mean and the variance of the distribution as it changes:

```python continuation
from skcomplex.core.components.laws.utils import moment
from skcomplex.core.specs import ObservableSpec


def mean_wealth(s):
    return moment(s.f, 1, classes, normalized=True)


def wealth_variance(s):
    return moment(s.f, 2, classes, central=True)


observables = [
    ObservableSpec('mean_wealth', mean_wealth, group='wealth'),
    ObservableSpec('wealth_variance', wealth_variance, group='wealth'),
]
trajectory = Simulation(economy, dt=10.0, observables=observables).prepare().run(until=100).trajectory_
variance = trajectory.values('wealth_variance')
round(float(trajectory.values('mean_wealth')[-1]), 3), variance[[0, -1]].round(3).tolist()
```

`moment(expression, order, coordinates)` is the reduction `sum_j c_j^r f_j` over the members of a group, with the
coordinate `c_j` of every member, such as its activity.

- The order-0 moment is the density.
- `normalized=True` divides by the density, so the normalized order-1 moment is the mean activity.
- `central=True` takes the moment about the mean, so the central order-2 moment is the variance.

Moments are expressions of `total` with a shared row of weights, a vector with one weight per reduced member. So laws
read them and observables record them. An observable of a group receives the namespace of one member, and its function
must be defined at module level so that it pickles.

#### The particle solver

Run the same declarations with individual particles instead of densities, and compare:

```python continuation
from skcomplex.core.engine.solvers import KineticSolver

sampled = Simulation(economy, dt=10.0, solver=KineticSolver(n=20_000), random_state=0).prepare().run(until=100)
round(float(np.abs(sampled.trajectory_.values('wealth.f')[-1] - densities[-1]).sum()), 2)
PassCompiler(solver=KineticSolver()).prepare(welfare).report_.ok
```

`KineticSolver(n, dt)` runs the declarations by direct simulation Monte Carlo. It holds every density as counts of
particles of one mass, with `n` particles for the largest initial density. In every step:

- a candidate particle encounters with the probability `1 - exp(-lambda_h dt)`, where `lambda_h = sum_k eta[h, k] g_k`,
- it picks its field activity and outcome in proportion to `eta[h, k] A[h, k, i] g_k`,
- a proliferation adds a Poisson number of particles, and a destruction removes a binomial number.

The state holds the empirical densities, so trajectories match the ODE form, and the error against the ODE solution
falls as one over the square root of `n`. The solver supports continuous time, lattice spaces, site groups, and
encounter laws only. Validation reports a rate law with it and lists the solvers that run it.

#### Levels

Move between individual agents and their distribution, so one level of a system can feed another:

```python continuation
from skcomplex.core.analysis import CoarseGrainAnalysis


class TraderElement(Element):
    def __init__(self, label, *, w: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.w = w


profile = densities[-1] / densities[-1].sum()
wealths = classes[np.random.default_rng(0).choice(classes.size, size=1000, p=profile)]
traders = GroupElement('traders', element=TraderElement, size=1000, values={'w': wealths}, variable=True)
density = CoarseGrainAnalysis(over=classes).prepare(traders.values['w']).run().field_
binned = GroupElement('wealth', element=WealthDensityElement, size=classes.size, on='sites', values={'f': density})
round(float(density.sum()), 6), round(float(density @ classes), 3), round(float(traders.values['w'].mean()), 3)
```

- `CoarseGrainAnalysis(over=values)` coarse-grains in state space. It bins the values of a state of a group's members,
  from their initial values or from a frame of a trajectory, over given values. Every member counts at the nearest
  value, so the densities sum to 1 and start a distribution site group.
- The other way round, members drawn from a distribution are values sampled with NumPy in proportion to the densities,
  and the group's constructor builds them.
- A system can then run one level in a partition of a co-simulation and feed another level its moments.

The library has `kinetic_wealth_welfare`, `immune_competition`, `kinetic_opinion`, `evacuation`, and `criminality`.
`evacuation` holds a crowd on a network of rooms as a site group on a two-dimensional lattice indexed by room and speed
class, whose walking encounters read the doors of the building.

#### Automata on groups

Run a cellular automaton on the elements of a group instead of a square grid, such as a free group, whose Cayley graph
is a tree. A majority rule on the ball of radius 4 of the free group on two generators:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_gap import PermutationGroup, libgap
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import CayleySpace, GeneratorNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec, TotalisticSpec

tree = CayleySpace(libgap.FreeGroup(['a', 'b']), ball=4).prepare()
memory = GeneratorNeighborhood(tree.generators_)


class MajorityElement(Element):
    def __init__(self, label, *, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive

    def _get_laws(self, s):
        return [LocalRuleLaw(s.alive, memory, TotalisticSpec({3: 1, 4: 1, 5: 1}))]


cells = (np.random.default_rng(0).random(tree.size_) < 0.5).astype(float)
majority = System(elements=[LatticeElement('cells', element=MajorityElement, space=tree, values={'alive': cells})])
tree.ball_sizes_, Simulation(majority, dt=1).prepare().run(until=3).trajectory_.values('cells.alive').shape
CayleySpace(PermutationGroup([[(1, 2)], [(1, 2, 3)]])).prepare().words_
```

A cellular automaton needs only a group. Its cells are the elements, and a cell at `g` reads the cells at `g s` for the
elements `s` of a finite memory set (Ceccherini-Silberstein and Coornaert 2023).

- `CayleySpace(group, ball=r)` is the Cayley graph of a finitely generated group on the ball of radius `r` in its
  generators and their inverses.
- `CayleySpace(group)` without a ball is the whole of a finite group, such as a finite quotient, where configurations
  are periodic.
- A group space is a space with sites, so `LocalRuleLaw`, `SampleLaw`, and the other lattice laws run on it with the
  NumPy and Numba backends, and a run reads only its neighbour table.
- Preparing the space enumerates its sites, numbered by word length. `generators_` names the generators, `words_` gives
  the shortest word of every site, and `ball_sizes_` gives the growth of the balls. Cells outside a ball read the
  space's `fill`.
- The memory set is a `MemoryNeighborhood` of words such as `'a'`, `'a^-1'`, or `'a*b'`, with `'1'` for the cell
  itself. `GeneratorNeighborhood(generators)` gives the identity followed by every generator and its inverse.

The group is a native Sage or GAP group. So any presentation works whose word problem a library solves:

- A free group, `libgap.FreeGroup(['a', 'b'])`, reduces its words directly. A finitely presented group,
  `free / relators`, is solved by Knuth-Bendix completion with KBMAG in GAP. A presentation it cannot complete raises
  `SpaceError`, which asks for a finite quotient.
- An automaton group of the AutomGrp package, which `libgap.LoadPackage('automgrp')` loads, has its elements told apart
  by their action on a level of the tree. An example is `libgap.AutomatonGroup('t = (1, t)(1, 2)')` for the adding
  machine, and `libgap.PermGroupOnLevel(group, n)` is its finite quotient on level `n`.
- A Sage `PermutationGroup` or `MatrixGroup` compares its elements directly.

GAP comes with the `passagemath-gap` dependency, and starts on the first ball that needs it. A ball whose size bound
exceeds `max_sites`, 100,000 by default, raises `SpaceTooLargeError` before GAP is called. The compiler reports it as
`space-too-large`, with the largest radius that fits.

A group space lacks the regular geometry of a square lattice and an image layout. So validation reports a checkerboard
schedule as `schedule-unsupported` and an image visual as `visual-unsupported`. Groups on group spaces stream to Rerun
with `CayleyVisual`: nodes on rings by word length, colored by state, with the edges of the generators logged once.

The core names no specific group, so you write a named group in one of these forms. The Grigorchuk group acts on the
binary tree by `a = (1, 1)(1, 2)`, `b = (a, c)`, `c = (a, d)`, `d = (1, b)`, so it is an automaton group. It is not
finitely presented, and its balls grow faster than any polynomial and slower than any exponential. Its action on
level 3 of the tree is a finite quotient:

```python
from sage.all__sagemath_gap import libgap

from skcomplex.core.components.spaces import CayleySpace

libgap.LoadPackage('automgrp')
grigorchuk = libgap.AutomatonGroup('a = (1, 1)(1, 2), b = (a, c), c = (a, d), d = (1, b)')
ball = CayleySpace(grigorchuk, ball=6).prepare()
quotient = CayleySpace(libgap.PermGroupOnLevel(grigorchuk, 3)).prepare()
ball.ball_sizes_, quotient.size_
```

**Theory of automata.** Ask whether an automaton is injective or surjective, find a pattern no configuration reaches,
or invert a reversible rule:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.analysis import AutomatonAnalysis
from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.laws import LocalRuleLaw
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace, MooreNeighborhood
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import OuterTotalisticSpec, StateSpec, TableSpec


class CellElement(Element):
    def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive


def automaton(space, neighborhood, rule):
    cells = LatticeElement('cells', element=CellElement, space=space)
    interaction = LocalRuleInteraction('rule', target='cells', state='alive', rule=rule, neighborhood=neighborhood)
    return System(elements=[cells], interactions=[interaction])


life = automaton(LatticeSpace((6, 6)), MooreNeighborhood(), OuterTotalisticSpec({(0, 3): 1, (1, 2): 1, (1, 3): 1}))
AutomatonAnalysis('surjective').prepare(life).run().surjective_
AutomatonAnalysis('garden-of-eden', random_state=0).prepare(life).run().pattern_ is not None
ring = LatticeSpace(8)
shift = automaton(ring, LatticeNeighborhood(((1,),)), TableSpec({(1,): 1}))
back = AutomatonAnalysis('inverse').prepare(shift).run().inverse_
composition = AutomatonAnalysis('compose', second=back).prepare(shift).run().composition_.interactions[0]
law = LocalRuleLaw(var('alive'), composition.neighborhood, composition.rule).prepare()
law.run({'space': ring, 'alive': [1, 0, 1, 1, 0, 0, 0, 1]}).value_.tolist()

step = LocalRuleLaw(var('alive'), shift.interactions[0].neighborhood, shift.interactions[0].rule).prepare()
cells = [1, 0, 1, 1, 0, 0, 0, 0]
for _ in range(2):
    cells = step.run({'space': ring, 'alive': cells}).value_
start = LatticeElement('cells', element=CellElement, space=ring, values={'alive': [1, 0, 1, 1, 0, 0, 0, 0]})
started = System(elements=[start], interactions=shift.interactions)
frames = Simulation(started, dt=1).prepare().run(until=2).trajectory_.values('cells.alive')
cells.tolist() == frames[-1].astype(int).tolist()
```

`skcomplex.core.analysis` decides the properties the theory studies on finite universes, where they are decidable:
periodic configurations on a torus or a finite quotient, and patterns on windows. The checks do not decide them on
infinite groups. A cellular automaton is a `System` with a `LocalRuleInteraction`, and `AutomatonAnalysis` prepares on
the system. The `run` of a `LocalRuleLaw` is one step of the map on configurations, so several steps are a loop over
that `run` or a `Simulation` of the system. The two give the same configuration, as the last line shows.

- The kinds `'injective'` and `'surjective'`, which agree on a finite universe, enumerate the configurations of small
  universes and solve a SAT encoding with PySAT on larger ones.
- `'pre-injective'` asks whether two configurations that differ only inside a `window` can share their image.
- `'garden-of-eden'` finds a pattern without a preimage, by enumeration or by SAT on random patterns.
- `'inverse'` finds the rule of the inverse of a reversible automaton with the smallest memory up to `max_radius`.
- `'compose'` gives the automaton of its composition with a `second` one.

**Linear automata and subshifts.** Decide a linear rule by algebra, and restrict configurations to those without a
forbidden pattern, such as the golden mean shift, where no two neighbouring cells are both 1:

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.analysis import AutomatonAnalysis
from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import LocalRuleInteraction
from skcomplex.core.components.spaces import LatticeNeighborhood, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import LinearRuleSpec, StateSpec, SubshiftSpec, TableSpec


class CellElement(Element):
    def __init__(self, label, alive: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0):
        super().__init__(label)
        self.alive = alive


class GoldenCellElement(Element):
    def __init__(
        self,
        label,
        alive: Annotated[float, StateSpec(domain=SubshiftSpec(LatticeNeighborhood(((0,), (1,))), ((1, 1),)))] = 0,
    ):
        super().__init__(label)
        self.alive = alive


def golden(space):
    return System(elements=[LatticeElement('cells', element=GoldenCellElement, space=space)])


laplacian = System(
    elements=[LatticeElement('cells', element=CellElement, space=LatticeSpace(5))],
    interactions=[
        LocalRuleInteraction(
            'laplacian',
            target='cells',
            state='alive',
            rule=LinearRuleSpec((1, -2, 1), 'GF(2)'),
            neighborhood=LatticeNeighborhood(((-1,), (0,), (1,))),
        ),
    ],
)
AutomatonAnalysis('injective').prepare(laplacian).run().injective_
AutomatonAnalysis('surjective', infinite=True).prepare(laplacian).run().surjective_
[
    AutomatonAnalysis('count').prepare(golden(LatticeSpace(length, boundary='fixed'))).run().count_
    for length in range(1, 8)
]
windows = [LatticeSpace(24, boundary='fixed')]
system = golden(windows[0])
AutomatonAnalysis('entropy', windows=windows).prepare(system).run().entropy_.round(3)

forbidden = [1, 1, 0, 0, 0]
cells = LatticeElement('cells', element=GoldenCellElement, space=LatticeSpace(5), values={'alive': forbidden})
keep = LocalRuleInteraction(
    'keep',
    target='cells',
    state='alive',
    rule=TableSpec({(1,): 1}),
    neighborhood=LatticeNeighborhood(((0,),)),
)
held = System(elements=[cells], interactions=[keep])
AutomatonAnalysis('admissible', configuration=forbidden).prepare(held).run().admissible_
Simulation(held, dt=1).prepare().run(until=2).trajectory_.values('cells.alive')[-1].astype(int).tolist()
```

The rule `LinearRuleSpec(coefficients, field)` is the element `sum_s a_s s` of the group ring over `'GF(p)'` or
`'QQ'`.

- On a finite universe it is a matrix, whose rank decides injectivity, surjectivity, and pre-injectivity together.
- On `Z^d`, with `infinite=True`, its Laurent polynomial decides them. It is invertible when the polynomial is a
  monomial, and surjective when it is not zero.
- The kinds `'matrix'` and `'polynomial'` of `AutomatonAnalysis` give them.

`SubshiftSpec(neighborhood, forbidden)` is the domain of a state whose configurations hold no forbidden pattern.
Admissibility is an analysis kind, and a run does not enforce it. A run checks only that every cell holds a value of
the alphabet, so it runs from a configuration with a forbidden pattern, as the last line shows.

- The kinds `'admissible'` and `'violations'` check a configuration.
- `'count'` counts the admissible configurations of a window, and `'sample'` samples them by a Markov chain of
  single-cell moves.
- `AutomatonAnalysis('entropy', windows=windows)` gives `log |X_F| / |F|` along windows `F`.
- `AutomatonAnalysis('difference')` gives the cells where two runs differ after one cell changes.
- `EntropyAnalysis('rate')` of the frames of a run as columns gives the spatial entropy rate of every frame, which never
  increases under an automaton.

The library has `free_group_automaton`, `lamplighter_automaton`, `heisenberg_automaton`, `baumslag_solitar_automaton`,
`grigorchuk_automaton`, the Laplacian `linear_ring_automaton`, and the golden mean shift `subshifts`, under the field
path `computer_science/automata`. Their groups are written in the forms above.

### Agent-based systems

Write the behaviour of individual agents as plain Python, and let them move, eat, reproduce, and die on a grid, a
network, or a continuous space. Foragers wander a torus, find food, split when they are full, and starve when they are
empty:

```python
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


class ForagerElement(Element):
    def __init__(
        self,
        label,
        visual=VisualSpec('agents', {'color_by': 'food'}),
        *,
        food: Annotated[float, StateSpec()] = 5.0,
        hunger: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ):
        super().__init__(label, visual=visual)
        self.food = food
        self.hunger = hunger

    def _get_laws(self, s):
        return [RuleLaw(self.forage)]

    def forage(self, agent, system):
        cells = agent.neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])
        agent.food += system.rng.poisson(0.8) - agent.hunger
        if agent.food <= 0:
            agent.remove()
        elif agent.food > 10:
            agent.food /= 2
            agent.create(food=agent.food)


foragers = System(
    elements=[GroupElement('foragers', element=ForagerElement, size=50, variable=True)],
    space=LatticeSpace((30, 30), boundary='periodic'),
)
trajectory = Simulation(foragers, random_state=0).prepare().run(until=20).trajectory_
int(trajectory.values('foragers.count')[-1]), len(trajectory.agents('foragers', 20).ids)
```

Agents are ordinary elements whose laws are rules written as code.

- An `Element` declares the states every agent holds, and the parameters its group shares, as annotated arguments of
  its `__init__`.
- Its `_get_laws` returns `RuleLaw`s, whose functions take the agent and the system.
- An element or a group whose laws include rules is an agent, and so is every member of a variable group.
- In a system with rules, a group whose element type has no laws is a group of passive agents that rules read and
  change, unless an interaction whose laws are not rules binds it.

A system with rules is agent-based. It runs in discrete time on `MesaSolver`, one Mesa step per unit of time, with
every agent activated in random order. `MesaSolver` declares what it runs like any solver, so
`PassCompiler(solver=MesaSolver()).prepare(system).report_` holds the laws, spaces, and times it does not support.

What an agent can do:

- An agent reads and sets its states as attributes, and reads its group's parameters, which change live like any
  parameter.
- `neighbors`, `neighbor_cells`, `agents_here`, `move_to`, `create`, and `remove` act on its place and its population.
  Only the members of a group declared `variable=True` are created and removed during a run.
- `agent.displacement_to(other)` gives the vector to another agent or a location, the shortest one on a torus, and
  `agent.distance_to(other)` its length, or the number of hops on a network.
- `agent.group` is the path of the agent's group, and `agent.group_label` its label.

Where agents live:

- `GroupElement(label, element=..., size=..., location=...)` places the agents at cell coordinates, network nodes, or
  positions, one per agent or one shared. Agents without a location take random places.
- `LatticeSpace` gives agents Moore or von Neumann neighbourhoods and an optional cell capacity, `NetworkSpace` places
  agents on the nodes of a NetworkX graph, and `ContinuousSpace` finds neighbours within a radius.
- `system.dimensions` gives the extent of the space per axis, and `system.torus` whether it wraps.

What the system gives a rule:

- the random generator `system.rng`, the other agents with `system.agents(name)` and `system.count(name)`, and the
  empty cells.
- Every rule receives the system view scoped to the interaction or agent group that owns it.
  `system.parameter('name')` reads the owner's parameter. `system.agents('role')` reads the group bound to a role of the
  owner, then a group by its path relative to the owner's system, then by its path from the root. So a rule works
  unchanged in a subsystem.
- Interactions with agent rules bind any number of agent groups, of any sizes, through their roles.

What a run records: frames and trajectories hold every group's agents with their ids, states, and locations, through
`trajectory.agents(group, frame)`, and the count of every group as the reporter `<group>.count`.

The catalog has Schelling segregation as `schelling`, the voter system as `voter_system`, Boltzmann wealth exchange as
`boltzmann_wealth`, and the wolf and sheep system as `wolf_sheep`.

An interaction can hold a rule too, and read the groups bound to its roles. Here walkers drift along a periodic road
inside a subsystem, and an interaction counts the pairs closer than its `reach`:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec


class WalkerElement(Element):
    def _get_laws(self, s):
        return [RuleLaw(self.step)]

    def step(self, agent, system):
        width, height = system.dimensions
        x, y = agent.location
        agent.move_to(((x + system.rng.uniform(0, 2)) % width, y))


class SpreadInteraction(Interaction):
    def __init__(self, label, *, walkers: Annotated[str, RoleSpec()], reach: Annotated[float, ParameterSpec()] = 10.0):
        super().__init__(label)
        self.walkers = walkers
        self.reach = reach

    def _get_laws(self, s):
        return [RuleLaw(self.measure)]

    def measure(self, system):
        walkers = system.agents('walkers')
        close = [a for a in walkers for b in walkers if a != b and a.distance_to(b) < system.parameter('reach')]
        counts.append(len(close) // 2)


counts = []
places = [(10.0 * index, 0.5) for index in range(5)]
road = System(
    systems=[
        System(
            'road',
            elements=[GroupElement('walkers', element=WalkerElement, size=5, location=places)],
            interactions=[SpreadInteraction('spread', walkers='walkers')],
        ),
    ],
    space=ContinuousSpace(extent=(50.0, 1.0), boundary='periodic'),
)
Simulation(road, random_state=0).prepare().run(until=5)
counts
```

#### Phases, markets, institutions, and learning

Order each step into phases, match buyers with sellers in a market, let an institution set policy, and let agents learn
from their past. Traders quote prices in one phase and trade in a batch auction in the next, while a clerk sets a fee
from the volume traded:

```python
from typing import Annotated, ClassVar

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import FeedbackInteraction, MatchingInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import BatchMatchingSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec


class TraderElement(Element):
    def __init__(
        self,
        label,
        *,
        value: Annotated[float, StateSpec()] = 0.0,
        bid: Annotated[float, StateSpec()] = 0.0,
        ask: Annotated[float, StateSpec()] = 0.0,
        demand: Annotated[float, StateSpec()] = 0.0,
        supply: Annotated[float, StateSpec()] = 0.0,
        money: Annotated[float, StateSpec()] = 0.0,
        goods: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label)
        self.value = value
        self.bid = bid
        self.ask = ask
        self.demand = demand
        self.supply = supply
        self.money = money
        self.goods = goods

    def _get_laws(self, s):
        return [RuleLaw(self.quote, phase='quote')]

    def quote(self, agent, system):
        agent.bid = system.rng.uniform(0, agent.value) if agent.demand else 0.0
        agent.ask = system.rng.uniform(agent.value, 2.0) if agent.supply else 2.0


class ClerkInteraction(FeedbackInteraction):
    reports: ClassVar[tuple[str, ...]] = ('traded',)

    def __init__(self, label, *, rule=None, phase=None, fee: Annotated[float, ParameterSpec()] = 0.0):
        super().__init__(label, rule=rule, phase=phase)
        self.fee = fee

    def act(self, system):
        traded = sum(agent.goods for agent in system.agents('buyers'))
        system.report('traded', traded)
        system.set_parameter('fee', 0.01 * traded)


rng = np.random.default_rng(0)
buyers = {'value': rng.uniform(0.5, 2, 20), 'demand': 1.0, 'money': 2.0}
sellers = {'value': rng.uniform(0, 1.5, 20), 'supply': 1.0}
auction = System(
    elements=[
        GroupElement('buyers', element=TraderElement, size=20, values=buyers),
        GroupElement('sellers', element=TraderElement, size=20, values=sellers),
    ],
    interactions=[
        MatchingInteraction(
            'auction',
            requesters='buyers',
            providers='sellers',
            rule=BatchMatchingSpec(),
            limit=('bid', 'ask'),
            quantity=('demand', 'supply'),
            allowance='money',
            transfers=(('money', -1), ('goods', 1)),
            phase='trade',
        ),
        ClerkInteraction('clerk', phase='trade'),
    ],
    time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('quote'), PhaseSpec('trade')])),
)
trajectory = Simulation(auction, random_state=0).prepare().run(until=5).trajectory_
prices = [round(float(price), 3) for price in trajectory.values('auction.value')[1:]]
prices, float(trajectory.values('clerk.traded')[-1]), round(float(trajectory.values('clerk.fee')[-1]), 3)
trajectory.values('auction.volume').tolist()
```

**Phases.** `DiscreteTime(schedule=PhasesScheduleSpec([...]))` lists the phases of a step.

- A `PhaseSpec(name, activation, repeat)` runs its agents `'random'`ly, `'sequential'`ly in order of creation,
  `'simultaneous'`ly, or `'by-type'`, for `repeat` rounds, such as intraday rounds.
- In a simultaneous phase every rule reads the states of the start of the phase, and the writes apply at its end.
- The activation `'visits'` runs `visits` times as many visits as the agents of `groups` at the start of the phase.
  Each visit goes to an agent drawn with replacement, which runs its rules of the phase, its element's rules and then
  its member rules, at the time `k + i / n` of visit `i` of `n` in step `k`.
- `RuleLaw(function, phase=name)` puts a rule in a phase. Rules without a phase run in the phase `'default'`, which runs
  first unless the system lists it. So systems without phases keep one random activation per step.

**Markets.** `MatchingInteraction(label, requesters=..., providers=..., rule=...)`, from
`skcomplex.core.components.interactions`, matches two groups.

- Requesters state the highest clearing value they accept in their `limit` state, the quantity they want in
  `quantity`, and the most value they can give in `allowance`. Providers state their lowest clearing value and the
  quantity they offer. A missing state counts as no limit, one unit, and no allowance.
- A name given as a pair, such as `limit=('bid', 'ask')`, names the state of the requesters and of the providers.
- The rule is a frozen spec of `skcomplex.core.specs` that names the matching algorithm and holds only its settings:
  - `RandomSearchSpec(tries, clearing)`, the default,
  - `PartnerSwitchSpec(sample, intensity, partner)`, partner choice with switching,
  - `QueueSpec(order)`,
  - `IterativeMatchingSpec(adjustment, iterations, tolerance)`, a clearing value adjusted until requests and offers
    balance,
  - `ContinuousMatchingSpec(arrival)`, where arriving offers match resting ones by limit and arrival,
  - `BatchMatchingSpec()`, one clearing value for all offers,
  - `StableMatchingSpec(requester_preference, provider_preference, clearing)`, stable matching by the `matching`
    library,
  - `PartnerSpec(candidates, choice, fallback, capacity, score, clearing)`, where requesters in random order pick among
    the candidates that still offer a quantity at a limit they accept.
- The rule may also be a function `rule(requesters, providers, rng, capacity)` that returns pairs, which match at the
  midpoint of both limits.
- The matching runs the algorithm of its rule on the offers of every round. Each match is a `MatchSpec`, whose `value`
  is the clearing value per unit.
- Every match lowers the quantities of both sides and moves the states of `transfers`. With `('money', -1)` the value
  times the quantity moves from the requester to the provider, and with `('goods', 1)` the quantity moves back, so the
  totals stay the same. `on_match(match, system)` adds any other effect.
- The matching reports `<label>.value`, the quantity-weighted clearing value, `<label>.volume`, and
  `<label>.unmatched`.

**Agent operations.** These interactions of `skcomplex.core.components.interactions` run as agent rules:

- `MovementInteraction` moves agents by neighbour, Gaussian, Lévy, best-cell, or user steps, with a condition, an
  acceptance, exclusion, and a cost.
- `TransferInteraction` moves amounts of states between members, the agents of one cell, a `Source()`, and a `Sink()`.
  It runs as transfer laws with `form='symbolic'`, or between paired agents with `form='agent'`.
- `LifeCycleInteraction` creates offspring by probability or rate, with inherited, split, and mutated states, and
  removes agents, the first or a random member of a group with `cull` and `pick`.
- `system.composition(group, state)` counts the agents of a group by the value of a state, kept up to date as agents
  change, so a birth probability can read the composition of the population.
- `GroupingInteraction` draws pairs, partitions, neighbourhoods, or groups by a state every step, and acts on each.
- `TopologyInteraction` adds and removes nodes and edges of a network, and rewires edges towards a weight.

**Institutions.** A `FeedbackInteraction`, from `skcomplex.core.components.interactions`, acts once per round of its
phase with its method `act`.

- It reads `system.observable(name)`, the reporters and the observables of the run at the end of earlier steps through
  `system.history(name, window)`, and the agents.
- It sets parameters of any element or interaction with `system.set_parameter(name, value)`. Rules see the change at
  once, and the trajectory records it.
- A subclass that declares its own parameters takes them as annotated arguments of its `__init__`, next to `rule` and
  `phase`.
- Elements and interactions declare the values they publish in the class variable `reports`, and publish them with
  `system.report(name, value)`. `system.storage` keeps private values of an element or an interaction across the steps
  of a run.

**Learning.** Agents learn from their own past and from each other.

- `agent.lags(state, window)`, a method of the agent view a rule receives, returns the past values of a state declared
  with lags, `StateSpec(lags=m)`. A rule writes adaptive, trend-following, anchoring, and fundamentalist forecasts from
  these values.
- In `skcomplex.core.components.interactions`, `ChoiceInteraction(form='agent')` draws an option by the logit,
  Metropolis, Fermi, or best rule, from a utility `utility(agent, system, option)` that can read `system.history` and
  `system.storage`.
- `ReinforcementInteraction` moves the scores of options towards their realized payoffs, or with `chosen` only the
  score of the option a member chose.
- Both run with `schedule='member'` in the activation of every member, such as one visit.
- `EvolutionInteraction` replaces strategies by selection, crossover, and mutation, with the system's random stream.
- `ImitationInteraction(rule='best')` copies the best neighbour.

#### Porting Mesa systems

Bring an existing Mesa model over, and gain live, validated, recorded parameters. A coin-flipping Mesa model becomes:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class CoinElement(Element):
    def __init__(
        self,
        label,
        *,
        heads: Annotated[float, StateSpec()] = 0.0,
        bias: Annotated[float, ParameterSpec()] = 0.5,
    ):
        super().__init__(label)
        self.heads = heads
        self.bias = bias

    def _get_laws(self, s):
        return [RuleLaw(self.flip)]

    def flip(self, agent, system):
        agent.heads += float(system.rng.random() < agent.bias)


coins = System('coins', elements=[GroupElement('coins', element=CoinElement, size=10)], time=DiscreteTime())
int(Simulation(coins, random_state=0).prepare().run(until=10).trajectory_.agents('coins', 10).values['heads'].sum())
```

A Mesa model is ported to a system piece by piece:

- its agent class becomes an element whose step is a `RuleLaw`,
- its agent attributes become states,
- the attributes of its `Model` become parameters, which then change live, are validated, and are recorded like those
  of any system.

Mesa still runs the steps, as the backend of agent-based systems.

### Cell complexes and link fields

Put states on the links and plaquettes of a lattice, or on the edges and faces of a network, as lattice gauge theories
and discrete exterior calculus do. A `CellComplexSpace` holds the sites, links, and plaquettes of a lattice, or the
nodes, edges, and faces of a network, with the oriented incidence between them. It is a `NetworkSpace`, so groups on
its `'edges'` and `'hyperedges'` hold states on links and plaquettes.

#### Cells of lattices and networks

Build the complex of a torus and of a tetrahedron, and read how their cells connect:

```python
import networkx as nx

from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace, NetworkSpace

torus = CellComplexSpace(LatticeSpace((8, 8))).prepare()
[matrix.shape for matrix in torus.incidences_], (torus.incidences_[1] @ torus.incidences_[0]).count_nonzero()

tetrahedron = NetworkSpace(nx.complete_graph(4))
cliques = CellComplexSpace(tetrahedron, faces='cliques', dimension=3).prepare()
cliques.incidences_[2].shape
```

- The links of a lattice point along the positive direction of their axis. Fixed and reflecting edges drop the cells
  that leave the box.
- A network takes `faces=None` for a minimum cycle basis, `'hyperedges'` for its hyperedges, `'cliques'` for its
  cliques up to `dimension`, or a sequence of node cycles.
- `incidences_[k]` has one row per cell of degree `k + 1` and one column per cell of degree `k`, so the product of two
  consecutive ones is 0.

#### Exterior calculus

Write field equations on a complex with the operators of discrete exterior calculus. Here a potential on links follows
lattice Maxwell equations, and their Gauss law holds to round-off:

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import codifferential, exterior_derivative
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class FieldLinkElement(Element):
    def __init__(
        self,
        label,
        visual=None,
        *,
        A: Annotated[float, StateSpec()] = 0.0,
        E: Annotated[float, StateSpec()] = 0.0,
    ):
        super().__init__(label, visual=visual)
        self.A = A
        self.E = E


space = CellComplexSpace(LatticeSpace((8, 8))).prepare()
potential = np.random.default_rng(0).normal(size=128)
links = GroupElement('links', element=FieldLinkElement, space=space, on='edges', values={'A': potential})
interactions = [
    RateInteraction('potential', target='links', state='A', rate=lambda n: -n.target.E),
    RateInteraction('field', target='links', state='E', rate=lambda n: codifferential(exterior_derivative(n.target.A))),
]
system = System(elements=[links], interactions=interactions)
trajectory = Simulation(system, dt=0.1).prepare().run(until=2.0).trajectory_
bool(np.abs(space.incidences_[0].T @ trajectory.values('links.E').T).max() < 1e-10)
```

The law utilities `exterior_derivative`, `codifferential`, and `hodge_star` act on the states of groups on the cells of
a complex. The degree of their argument is the part its group lives on.

- `exterior_derivative` maps to the next degree, and applied twice it is exactly 0.
- `codifferential` is `(-1)^k star^-1 d^T star` on a state of degree `k`. On an edge state it is the outflow minus the
  inflow at every site, and on a face state it is the curl on every edge. Applied twice it is 0 up to round-off.
- `hodge_star` is a diagonal factor. On a lattice it is the ratio of dual to primal cell volumes, and on a network it is
  the parameter `weight` of the group on that part of the complex.

With this sign, `codifferential(exterior_derivative(x))` is the graph Laplacian of a site state, and the curl of the
curl of an edge state. A rate of `RateInteraction` reads another group on the complex through `sources`, and the value
must have the degree of the state it changes.

#### Coface reductions

Sum over the faces around an edge, or the edges around a site. `neighbors(expression, over=CofaceNeighborhood())`
reduces over the faces that contain an edge, or the edges that contain a site. Inside the reduction,
`exterior_derivative` is the oriented sum around the one coface, which is the holonomy of an abelian connection, and
every coface counts with weight one. The next example uses it for the Wilson action.

#### Sampling links

Sample a lattice gauge field at equilibrium, such as compact U(1) with the Wilson action, link by link:

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, pi

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.laws.utils import exterior_derivative, neighbors
from skcomplex.core.components.spaces import CellComplexSpace, CofaceNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import StateSpec


class AngleLinkElement(Element):
    def __init__(self, label, visual=None, *, theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0):
        super().__init__(label, visual=visual)
        self.theta = theta


space = CellComplexSpace(LatticeSpace((8, 8))).prepare()
links = GroupElement('links', element=AngleLinkElement, space=space, on='edges')
sampling = SamplingInteraction(
    'wilson',
    target='links',
    state='theta',
    objective=lambda n: -2.0 * neighbors(cos(exterior_derivative(n.target.theta)), over=CofaceNeighborhood()),
    move='rotate',
    width=1.5,
    schedule='colored',
)
system = System(elements=[links], interactions=[sampling], time=DiscreteTime())
states = Simulation(system, random_state=0).prepare().run(until=200).trajectory_.values('links.theta')
round(float(np.cos(space.incidences_[1] @ states[-50:].T).mean()), 1)
```

The last line is the mean plaquette, the average cosine of the oriented sum of the angles around every face, which
lies between 0.6 and 0.8 here.

- `SamplingInteraction` samples a group on the edges of a cell complex with a local objective over cofaces.
- Its `'colored'` schedule updates together only links that share no face, and every draw comes from the generator of
  the simulation.
- The move is `'rotate'` for U(1), and `'random-value'` for Z_N with the `'heat-bath'` rule.

#### Analysing complexes

Split a link field into its gradient, curl, and harmonic parts, count the holes of a complex, or measure how curved a
gauge field is:

```python
import numpy as np

from skcomplex.core.analysis import CellComplexAnalysis
from skcomplex.core.components.spaces import CellComplexSpace, LatticeSpace

torus = CellComplexSpace(LatticeSpace((8, 8))).prepare()
uniform = np.where(torus.link_axes_ == 0, 1.0, 0.0)
hodge = CellComplexAnalysis('hodge', space=torus).prepare(uniform).run()
CellComplexAnalysis('betti', space=torus).prepare().run().betti_, bool(np.allclose(hodge.harmonic_[0], uniform))
```

A uniform flow around the torus is purely harmonic, and the torus has one component, two independent loops, and one
cavity. `CellComplexAnalysis` reads a state of a group on a complex from a trajectory, or an array with the complex as
`space`.

- The kind `'hodge'` sets `gradient_`, `curl_`, and `harmonic_`, which are orthogonal and sum to the state, and the site
  potential `potential_`.
- The kind `'betti'` sets `betti_`.
- The kind `'curvature'` sets `share_`, the share of faces whose oriented sum of an angle state is not a multiple of its
  period, and `energy_`, the sum of one minus the cosine of the sums.

`CellVisual`, named `VisualSpec('cell')`, draws edge states as colored segments. It draws their face sums and
hyperedge states as an image on a lattice complex, or as polygons on a network complex, with a cyclic palette for angle
states.

### Rewriting systems

Grow a structure generation by generation, as an L-system grows a plant or a substitution tiling grows a fractal. A
rewriting system replaces its parts in parallel at every step.

```python
from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import RewritingCompiler
from skcomplex.core.engine.runtime import Simulation


class AlgaeElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


algae = System(elements=[AlgaeElement('algae', axiom='A')])
trajectory = Simulation(algae, compiler=RewritingCompiler()).prepare().run(until=5).trajectory_
trajectory.values('algae.size').tolist(), trajectory.structure('algae', 5).text
```

There are two kinds of rewritten element:

- A `WordElement` is a string of symbols that grows by string productions, in the style of L-systems.
- A `PatchElement` is a set of tiles that grows by affine substitutions.

The system compiles to the rewriting form, and every step is one generation. `run(until=n)` records the generations `0`
to `n`. The structure itself lives next to the state vector. The states are the count of every symbol or prototile
that is a word, and the size `<element>.size`. `Trajectory.structure(path, frame)` returns the recorded structure of a
frame, which later generations do not change.

#### Words and productions

Write the rules of a word as productions, and every generation applies them all at once. A word or a patch rewrites
itself by the rewriting laws it returns from `_get_laws`, a `ProductionLaw` for every rule of a word. Strings are words
of one-character symbols, and `ModuleSpec` gives a symbol parameters.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import RewritingCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ModuleSpec, ParameterSpec

x = var('x')


class ShrinkingElement(WordElement):
    def __init__(self, label, axiom=(), *, alphabet=None, ratio: Annotated[float, ParameterSpec()] = 0.5):
        super().__init__(label, axiom, alphabet=alphabet)
        self.ratio = ratio

    def _get_laws(self, s):
        successor = [ModuleSpec('F', (x,)), ModuleSpec('A', (s.ratio * x,))]
        return [ProductionLaw(ModuleSpec('A', (x,)), successor, condition=x > 0.1)]


plant = System(elements=[ShrinkingElement('plant', axiom=[ModuleSpec('A', (1.0,))], alphabet={'A': 1, 'F': 1})])
Simulation(plant, compiler=RewritingCompiler()).prepare().run(until=5).trajectory_.structure('plant', 5).text
```

A production can take several forms:

- **Stochastic.** Its probabilities sum to one per predecessor and context, and it draws from the simulation's random
  stream.
- **Context-sensitive.** It depends on a left or a right context. The context skips the word's `ignore` symbols and
  treats bracketed branches as Prusinkiewicz and Lindenmayer (1990) do. A context-sensitive production takes
  precedence over context-free productions.
- **Parametric.** It binds the parameters of the predecessor to symbols, such as `x` above. Its condition and the
  parameters of its successor are expressions of these symbols and of the parameters of the rewritten element.
  `schedule` and `update_parameters` change those parameters between generations.

A word element with parameters declares them as annotated arguments of its `__init__`, as `ratio` above.

A generation that would make a word longer than `max_size` raises `RewritingError` before allocating it. The frames
before it stay recorded.

#### Word geometries, patches, and measures

Draw a word as a curve, or grow a patch of tiles, and measure the result as a fractal.

```python
from skcomplex.core.analysis import ScalingAnalysis
from skcomplex.core.components.elements import PatchElement, WordElement
from skcomplex.core.components.laws import ProductionLaw, SubstitutionLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import RewritingCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import AffineSpec, GeometrySpec, PrototileSpec


class KochElement(WordElement):
    def _get_laws(self, s):
        return [ProductionLaw('F', 'F+F--F+F')]


koch = System(elements=[KochElement('koch', axiom='F', geometry=GeometrySpec(angle=60.0, contraction=1 / 3))])
curve = Simulation(koch, compiler=RewritingCompiler()).prepare().run(until=4).trajectory_.structure('koch', 4)
len(curve.segments()), round(curve.length(), 6)
```

A `GeometrySpec`, from `skcomplex.core.specs`, draws a word as line segments in two or three dimensions, as the turtle
interpretation of L-systems does:

- `F` draws forward, and `f` moves forward without drawing.
- `+` and `-` turn by the angle.
- `[` and `]` push and pop a branch.
- `contraction` scales the step at every generation.

A `PatchElement` holds tiles. Each tile is a `PrototileSpec`, a segment or a polygon, under an affine placement. A
`SubstitutionLaw` replaces every tile of a prototile by pieces, each placed by an `AffineSpec` map relative to it. The
Cantor set below keeps the outer thirds of every interval.

```python continuation
interval = PrototileSpec('interval', [(0, 0), (1, 0)])
thirds = [('interval', AffineSpec(scale=1 / 3)), ('interval', AffineSpec(scale=1 / 3, translation=(2 / 3, 0)))]


class CantorElement(PatchElement):
    def _get_laws(self, s):
        return [SubstitutionLaw('interval', thirds)]


cantor = System(elements=[CantorElement('cantor', prototiles=[interval])])
trajectory = Simulation(cantor, compiler=RewritingCompiler()).prepare().run(until=5).trajectory_
trajectory.values('cantor.size')[-1], round(trajectory.values('cantor.length')[-1], 6)
grid = trajectory.structure('cantor', 5).raster(729)
round(ScalingAnalysis('box-counting', scales=[3, 9, 27]).prepare(grid).run().exponent_, 2)
```

Structures with a geometry report `<element>.length` and `<element>.area` at every frame. They also give `bounds()`, and
`raster(resolution)`, a boolean grid that `ScalingAnalysis('box-counting')` reads, as above.

The `geometry` visual logs segments and outlines as line strips, and a patch with `filled=True` as an image of its
raster. It draws within the bounds the element declares, or else within the bounds of its first tiles.

### Quantum systems

Build a quantum system, such as a chain of spins, and follow its state in time or find its spectrum. A quantum system is
a set of quantum subsystems whose state is one vector of their joint Hilbert space.

```python
from skcomplex.core.analysis import OperatorAnalysis
from skcomplex.systems import create_system

spectrum = OperatorAnalysis(n_states=2).prepare(create_system('quantum_heisenberg', n_sites=4)).run()
spectrum.energies_.round(6).tolist()
```

The core names this machinery for its mathematics, linear operator dynamics on tensor products, while the physics
systems keep the words of physics. For example, `GeneratorLaw` takes a Hermitian operator, which physics calls a
Hamiltonian.

Every subsystem holds its state `psi` in the Hilbert space of a `HilbertSpec`, which names its operators. You declare a
subsystem of your own with the operators it needs. Typical subsystems have these operators:

- A spin has the components `x`, `y`, `z` and the ladder operators `plus` and `minus` of spin `j`.
- A qubit has the Pauli matrices.
- A mode has `annihilate`, `create`, and `number`, truncated at `quanta`.
- A particle on the sites of a lattice or a network has its `laplacian` and its `adjacency`.

A group of subsystems, such as a `LatticeElement` of spins, is the tensor product of the spaces of its members, in
member order. The initial state of every subsystem is a basis label, an index, or a vector.

#### Operators and laws

Write the generator of a system from the operators of its subsystems, and add dissipation to open it.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components.elements import Element, LatticeElement
from skcomplex.core.components.interactions import NeighborInteraction
from skcomplex.core.components.laws import GeneratorLaw
from skcomplex.core.components.laws.utils import local_operator
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import OperatorCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import HilbertSpec, ObservableSpec, ParameterSpec, RoleSpec, StateSpec

PAULI = {
    'x': (np.array([[0, 0.5], [0.5, 0]]), True),
    'y': (np.array([[0, -0.5j], [0.5j, 0]]), True),
    'z': (np.array([[0.5, 0], [0, -0.5]]), True),
}
SPIN_HALF = HilbertSpec(2, PAULI, basis={'up': 0, 'down': 1})


class SpinElement(Element):
    def __init__(self, label, psi: Annotated[object, StateSpec(domain=CC)] = 'up', *, hilbert=SPIN_HALF):
        super().__init__(label)
        self.psi = psi
        self.hilbert = hilbert


class CouplingInteraction(NeighborInteraction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], J: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.J = J

    def _get_laws(self, s):
        a, b = s.source.psi, s.target.psi
        exchange = sum(local_operator(a, name) * local_operator(b, name) for name in ('x', 'y', 'z'))
        return [GeneratorLaw(s.J * exchange)]


def sz(member):
    return local_operator(member.psi, 'z')


def sites(values):
    return values


chain = System(
    elements=[LatticeElement('chain', element=SpinElement, space=LatticeSpace(6), values={'psi': ['up', 'down'] * 3})],
    interactions=[CouplingInteraction('coupling', target='chain')],
)
observables = [ObservableSpec('sz', sz, group='chain', reduce=sites)]
simulation = Simulation(chain, dt=0.5, observables=observables, compiler=OperatorCompiler())
trajectory = simulation.prepare().run(until=2.0).trajectory_
trajectory.values('sz').shape, round(float(np.ptp(trajectory.values('energy'))), 8)
```

Laws name the operators of a subsystem with `local_operator(s.psi, name)`, from `skcomplex.core.components.laws.utils`.
Operators combine by sums, numbers, and symbolic coefficients of parameters and time. `operator_product` and `dagger`
keep the order of products and adjoints. Two laws build the dynamics:

- `GeneratorLaw(operator)` adds to the generator, with `ħ = 1`.
- `DissipatorLaw(operator, rate)` adds a Lindblad dissipator, which makes the system open.

A `NeighborInteraction` is written once for a bond, with `s.source` and `s.target`. It acts on every pair of neighbours
of the space of the group that its one role binds. Compiling checks that the generator is Hermitian, and that the
dimension stays within the `max_dimension` of the operator solver.

A whole-system initial state, such as an entangled vector or `'ground'`, is given to the simulation as
`initial_state={'<system label>.state': value}`.

#### Solvers, observables, and spectra

Choose how the state evolves, record expectations of operators, and diagonalise the generator exactly. The first block
of this section finds the two lowest energies of a Heisenberg chain of the catalog.

A quantum system compiles to the operator form. Its state is stored as complex pairs under `<system>.state`, a pure
state for a closed system and a density matrix for an open one. Its solvers hand every frame to QuTiP:

- `UnitarySolver` solves the Schrödinger equation.
- `LindbladSolver`, the default, solves the master equation.
- `JumpTrajectoriesSolver` draws quantum jumps from the random stream of the run.

An `ObservableSpec` whose expression is an operator records its expectation. One over a group gives one value per
member when its `reduce` keeps the values, as `sites` does above. Every quantum run also records its `energy`, and its
`purity` when it is open. Groups on a lattice stream the space-time image of their local expectations to Rerun.

`OperatorAnalysis` analyses quantum systems by its kind:

- `'spectrum'` and `'ground-state'` diagonalise the generator exactly, optionally in the sector of a conserved diagonal
  operator.
- `'purity'` and `'entanglement'` read trajectories.

`SymbolicAnalysis('hamiltonian')` prints the Hamiltonian in operator notation.

### Stocks and flows

Build stocks that change only through flows, as in system dynamics, check that what flows is conserved, and find the
feedback loops that drive them. A stock is a state tagged with a quantity, such as `StateSpec(quantity='people')`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import Sink, Source, TransferLaw
from skcomplex.core.components.laws.utils import lookup
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec


def birth_rate(s):
    return s.b * s.n * lookup(s.n, [(0, 1.0), (100, 1.0), (200, 0.0)])


class PeopleElement(Element):
    def __init__(
        self,
        label,
        *,
        n: Annotated[float, StateSpec(quantity='people')] = 10.0,
        b: Annotated[float, ParameterSpec()] = 0.03,
        d: Annotated[float, ParameterSpec()] = 0.01,
    ):
        super().__init__(label)
        self.n = n
        self.b = b
        self.d = d

    def _get_laws(self, s):
        return [
            TransferLaw(Source(), s.n, birth_rate(s), label='births'),
            TransferLaw(s.n, Sink(), s.d * s.n, label='deaths'),
        ]


crowd = System(elements=[PeopleElement('crowd')])
observed = [ObservableSpec('birth_rate', birth_rate, element='crowd')]
trajectory = Simulation(crowd, dt=1.0, observables=observed).prepare().run(until=10).trajectory_
round(float(trajectory.values('crowd.births')[-1]), 3), round(float(trajectory.values('crowd.deaths')[-1]), 3)
round(float(trajectory.values('crowd.birth_rate')[-1]), 3)
```

A `TransferLaw(source, sink, amount, label=...)` moves an amount of the quantity from one stock to another. It exchanges
the quantity with the environment through `Source()` and `Sink()`. What the amount means depends on the laws around it:

- In a system of rate laws the amount is a rate.
- In a system of update laws it is moved once per step.
- With `rate=...` it is the size of a jump with that propensity.

Trajectories record every flow under `<element path>.<label>`, such as `crowd.births`.

An intermediate expression, such as the birth rate, is a function of the namespace that the laws call.
`ObservableSpec(label, function, element=element)` records the same function as the observable `<element>.<label>`,
which Rerun logs under the element. `lookup(x, points)` is a table function, piecewise linear between its points and
flat outside them.

#### Conservation and loops

Check that the stocks change only through flows, and see how much of each quantity enters from outside.

```python continuation
from skcomplex.core.analysis import LoopAnalysis, SymbolicAnalysis
from skcomplex.core.engine.runtime import ConservationObserver

SymbolicAnalysis('conservation').prepare(crowd).run().inflows_
monitored = Simulation(crowd, dt=1.0, observers=[ConservationObserver()]).prepare().run(until=10).trajectory_
LoopAnalysis('flow-matrix').prepare(monitored).run().matrix_.sum(axis=1).tolist()
```

- A rate or update law outside flows that changes a tagged state raises `ConservationError` when the system compiles.
- `SymbolicAnalysis('conservation')` returns the net inflow of every quantity from the environment, after checking that
  the internal flows cancel.
- The compiler accumulates that inflow in the hidden state `__boundary.<quantity>`. The observer
  `ConservationObserver` checks at every frame that each total changed only by it.
- `LoopAnalysis('flow-matrix', frame=frame)` gives the transaction-flow matrix of a frame of a trajectory, with one row
  per flow and one column per holder. Its rows sum to zero.

`distributed_delay(value, time, order, conserved=True)` is a material delay of any order. It is a chain of hidden stocks
in series, and it returns their outflow. With `conserved=False` it is an information smoothing. It adds hidden stocks
`<element path>.<kind>_delay<order>_<index>` to the rate laws of single elements and interactions, and they start in
equilibrium with their input. The system dynamics names `delay1`, `delay3`, and `smooth` of
`skcomplex.core.components.laws.utils` are its cases.

`LoopAnalysis('loops')` derives the signed dependency graph of a system from its laws, and lists its feedback loops with
their polarity. `LoopAnalysis('dominance')` ranks the loops by their gains at every frame of a trajectory.

```python continuation
analysis = LoopAnalysis().prepare(crowd).run()
[(polarity, ' -> '.join(nodes)) for nodes, polarity in zip(analysis.loops_, analysis.polarities_, strict=True)]
dominance = LoopAnalysis('dominance').prepare(trajectory).run()
dominance.polarities_[int(dominance.dominant_[-1])]
```

An edge of the graph goes from one state to another when the rate of the second depends on the first, with the sign of
the entry of the Jacobian. In systems that declare flows, the edges also go through the flows: from a state to every
flow it drives, and from a flow to its sink, with sign `+`, and to its source, with sign `-`.

The graph is that of any system with a vector field or a map, not only of stocks and flows. At one prey and one
predator, prey growth closes a reinforcing self-loop, predator death a balancing one, and predation a balancing loop
through both populations.

```python continuation
from skcomplex.systems import create_system

predation = LoopAnalysis(at=[1.0, 1.0]).prepare(create_system('lotka_volterra')).run()
dict(zip(predation.loops_, predation.polarities_, strict=True))
```

#### Conserved quantities between agents

Move money or goods between agents so that the totals stay constant, and let the run check it.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.laws.utils import transfer
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec, StateSpec


def net_worth(agent):
    return agent.cash + agent.claims - agent.equity


def lend(agent, system):
    borrower = system.agents('traders')[0]
    if borrower != agent:
        transfer(agent, borrower, 1.0, 'cash')
        transfer(borrower, agent, 1.0, 'claims')


class TraderElement(Element):
    def __init__(
        self,
        label,
        *,
        cash: Annotated[float, StateSpec(quantity='money')] = 10.0,
        claims: Annotated[float, StateSpec(quantity='claims')] = 0.0,
        equity: Annotated[float, StateSpec()] = 10.0,
    ):
        super().__init__(label)
        self.cash = cash
        self.claims = claims
        self.equity = equity

    def _get_laws(self, s):
        return [RuleLaw(lend)]


balance = ObservableSpec('net_worth', net_worth, group='traders', reduce='sum')
market = System(elements=[GroupElement('traders', element=TraderElement, size=3)])
simulation = Simulation(market, observables=[balance], invariants={'net_worth': 1e-9})
traders = simulation.prepare().run(until=3).trajectory_
float(traders.agents('traders', 3).values['cash'].sum())
```

Agents move conserved quantities between each other inside rules. `transfer(source, sink, amount, state)` moves an
amount between two agents at once. A negative holding is an amount owed, and total holdings stay constant.

A run checks that an expression keeps its first value with `Simulation(invariants={name: tolerance})`, where the name is
an observable of the run. At every frame the run compares the observable with its value at the first frame. It raises
`ConservationError`, naming the observable and the first frame that breaks it. Removing an agent that still holds a
tagged state also raises `ConservationError`, until its holdings are settled with transfers.

#### Translating systems

Turn a Vensim or Stella file into a system you can run and analyse. `SystemDynamicsTranslator(path)` reads a Vensim
(`.mdl`) or XMILE (`.xmile`, `.xml`, `.stmx`) file and writes the source of a system module.

```python
import importlib
import sys
import tempfile
from pathlib import Path

from skcomplex.core.analysis import LoopAnalysis
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import EulerSolver
from skcomplex.core.engine.translators import SystemDynamicsTranslator

text = """
Population = INTEG(Births - Deaths, 100) ~ People ~ |
Births = Population * Birth Fraction ~ People/Year ~ |
Deaths = Population / Lifetime ~ People/Year ~ |
Birth Fraction = 0.03 ~ 1/Year ~ |
Lifetime = 70 ~ Years ~ |
Doubling Share = Population / 200 ~ Dmnl ~ |
INITIAL TIME = 0 ~ Years ~ |
FINAL TIME = 100 ~ Years ~ |
TIME STEP = 0.25 ~ Years ~ |
SAVEPER = 1 ~ Years ~ |
"""
folder = Path(tempfile.mkdtemp())
_ = (folder / 'people.mdl').write_text(text)
translator = SystemDynamicsTranslator(folder / 'people.mdl').prepare().run()
_ = (folder / 'translated_people.py').write_text(translator.source_)
sys.path.insert(0, str(folder))
module = importlib.import_module('translated_people')
translator.settings_.until, module.DT

system = module.create_system()
simulation = Simulation(system, dt=module.SAVE_EVERY, observables=module.observables(system))
trajectory = simulation.prepare().run(until=module.UNTIL).trajectory_
trajectory.values('people.births')[0], trajectory.values('people.doubling_share')[0]
LoopAnalysis().prepare(system).run().polarities_
```

PySD reads the file. The translator writes `source_` from PySD's abstract model, and `settings_` from the control
section of the file. You save the source and import it like any module. The module defines:

- one element class, whose states are the stocks and whose parameters are the constants,
- the functions `create_system()` and `observables(system)`,
- the constants `START`, `UNTIL`, `DT`, and `SAVE_EVERY`.

Time in the module runs from zero, and `Time` in the file becomes `t + START`. The parts of the file become these:

- A flow variable that leaves one stock and enters another is a `TransferInteraction` between the two states. A flow
  variable of one stock is a `TransferInteraction` with `Source()` or `Sink()`. The trajectory reports each flow as
  `<element>.<flow>`.
- The rest of the net rate of a stock is a `RelaxationInteraction` when it has the form `(goal - stock) / time` or
  `(goal - stock) * rate`, and a `RateInteraction` otherwise.
- An auxiliary is a private function `_name(s, t)` that the laws call, and an `ObservableSpec` under its snake-case
  name.
- A lookup is a `lookup` table. The built-in functions are `where`, `step`, `ramp`, `pulse`, `pulse_train`, `delay1`,
  `delay3`, `smooth`, `distributed_delay`, `delayed`, `minimum`, `maximum`, `clip`, and the draws `uniform` and
  `normal`.
- A system whose stocks are all subscripted over one range is a `GroupElement` with one member per subscript. `SUM` over
  the range is `total`, and a read of one subscript is a `total` over that member.

The default solver finds the exact solution. Vensim and Stella integrate with Euler steps by default. A run with
`solver=EulerSolver(dt=module.DT)` and `dt=module.DT` reproduces their run step by step. A run with the default solver
differs from it by the Euler error of the time step of the file.

```python continuation
stepped = Simulation(system, solver=EulerSolver(dt=module.DT), dt=module.DT)
final = stepped.prepare().run(until=module.UNTIL).trajectory_.values('people.population')[-1]
round(float(final), 3)
```

One `ValidationError` lists every construct that does not translate, with its variable. These are `GAME`, `ALLOCATE`,
the vector functions, `TREND`, `FORECAST`, `SAMPLE IF TRUE`, data read from spreadsheets, macros, variables subscripted
over more than one range, systems that mix subscripted and unsubscripted stocks, delays and smoothings whose initial
value differs from their input, and `DELAY FIXED` of anything other than a stock.

The library sets these limits on what a translated module does:

- The core takes random draws only in update and renewal laws. A module with a random function translates, but raises
  `IncompatibleLawsError` when its system is prepared.
- A run cannot record an auxiliary that reads `DELAY FIXED`, or a delay that no law reads, so `observables` leaves it
  out.
- A trajectory records the flow of one member from the values of that member. So in a group, a flow variable that reads
  `SUM` or one named subscript stays in the rate of its stock, without a flow, and its value stays an observable.

The catalog systems built on flows are these:

- engineering processes: `bathtub`, `rice_milling`, `solid_waste`, `food_relief`, and `irrigation`,
- markets: `car_dealer`, `commodity_cycle`, and `beer_game`,
- innovation: `bass_diffusion`,
- macroeconomics: `stock_flow_consistent_economy`,
- organizations: `success_to_the_successful`, `shifting_the_burden`, and `escalation`.

### Linear programs

Set some states of a system to the optimum of a linear program, as flux balance analysis sets the fluxes of a cell.
`LinearProgramLaw` sets algebraic states, its variables, to an optimal solution, and HiGHS solves the program through
`highspy`, in the default install.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import LinearProgramLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import LinearObjectiveSpec, ParameterSpec, StateSpec


class RoutesElement(Element):
    def __init__(
        self,
        label,
        direct: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        detour: Annotated[float, StateSpec(algebraic=True)] = 0.0,
        capacity: Annotated[float, ParameterSpec()] = 3.0,
    ):
        super().__init__(label)
        self.direct = direct
        self.detour = detour
        self.capacity = capacity

    def _get_laws(self, s):
        constraints = [s.direct <= s.capacity, s.detour <= 4, s.direct >= 0, s.detour >= 0]
        objective = LinearObjectiveSpec(s.direct + s.detour)
        return [LinearProgramLaw([s.direct, s.detour], constraints, [objective])]


system = System(elements=[RoutesElement('flows')], time=StaticTime())
trajectory = Simulation(system).prepare().run(0.0).trajectory_
trajectory.values('flows.objective_0')[0]
```

The objective reaches 7, three on the direct route at its capacity and four on the detour. The rules of a program are
these:

- The constraints are Sage relations, `==`, `<=`, or `>=`, linear in the variables. A coefficient of a variable reads no
  state, and the other terms may read states, parameters, and the time. A relation in one variable is a bound of that
  variable.
- The objectives are `LinearObjectiveSpec`s, linear in the variables or a sum of their absolute values. They solve in
  order, and each optimum is kept within a relative 1e-9 while the next one solves.
- `active` is a condition. Where it is false the solve is skipped and the variables are zero.
- `infeasible` is `'raise'`, `'zero'`, or `'hold'`. `InfeasibleError` names the program, the time, the member, and the
  constraints and variables of an irreducible infeasible subset.
- A run records the variables under their paths, and the objective values under `<path>.objective_<k>`. With
  `duals=True` it also records the duals of the constraints under `<path>.dual_<k>`. The dual of a constraint is the
  change of the first objective with its right side.

#### Merged programs

The laws whose variables overlap, from any element or interaction, form one program. Their variables unite, their
constraints add, and at most one of them holds objectives. So an interaction adds a bound to the program of an element
by returning a law over the variable it bounds, and the tighter of two bounds holds.

#### Switches

A switch turns a part of a program on or off, such as a gene knock-out. It is a parameter that a bound reads, such as
`s.v <= 1000 * either(s.g1 > 0.5, s.g2 > 0.5)`. A knock-out sets the parameter to 0 in one of three ways:

- at construction,
- between runs, with `update_parameters`,
- for an adaptive parameter, by an `UpdateLaw` of an interaction that reads a Boolean node.

#### The forms of a program

A linear program law does not decide the form of the system that holds it. The program solves wherever the form needs
it:

- Beside rate laws under continuous time, the system keeps the ODE form, and the drift solves the program first at each
  evaluation. This is the direct approach. Its right side is only piecewise smooth, so use fixed steps, such as
  `RK4Solver`, rather than an adaptive integrator.
- Alone or beside update laws under discrete time, the system keeps the map form, and each step solves the program
  first.
- Alone under static time, the system has the static form, and each frame solves the program from the current
  parameters.
- In a discrete-time nested system beside a sibling with rate laws, the program solves at every synchronisation point of
  `CoSimulationSolver`, whatever the length of the interval, and the sibling reads its variables as held inputs. This is
  the static optimisation approach.
- In a group on the sites of a lattice, the program solves per member where `active` holds, each with its own warm
  start.

#### Issues of linear programs

`PassCompiler` reports these issues before it compiles:

- `linear-program-form`: a relation is not linear in the variables, or a coefficient of a variable reads a state.
- `linear-program-objectives`: two merged laws hold objectives.
- `linear-program-variable`: a variable is not declared with `StateSpec(algebraic=True)`.

A program under any other time gives the issue `law-time`.

#### Elementary modes

`ElementaryModeAnalysis` enumerates exactly the generators of the steady-state cone of a program, with the Parma
Polyhedra Library. The kind `'modes'` gives the elementary modes, and the kind `'pathways'` the extreme pathways. The
number of modes grows combinatorially, so the enumeration stops after `timeout` seconds with `EnumerationLimitError`.

#### Metabolism

The metabolic systems of the catalog, such as `flux_balance`, `dynamic_fba`, `community_fba`, and `comets`, are built
from the parts above. A metabolic network is an element whose data are its reactions. It has one algebraic state per
reaction, one adaptive switch per gene, and one lower and one upper bound parameter per reaction.

`elementary_flux_modes` runs on the default install. The systems built on the E. coli core model, which COBRApy bundles,
need the extra `metabolism`: `flux_balance`, `dynamic_fba`, `regulatory_fba`, `metabolic_perturbations`,
`community_fba`, `dynamic_multispecies_fba`, and `comets`. Install it with `pip install
'scikit-complexity[metabolism]'`. Without it they raise `MissingExtraError`.

The extra holds COBRApy, StrainDesign, and MICOM. COBRApy brings GLPK into the environment, which is under the GPL-3.0.
The package imports no module of GLPK, and it points COBRApy to the `hybrid` solver interface. The extra also holds
pandas below version 3, since COBRApy caps it.

#### Isotope labelling

Only the system `carbon_13_mfa` uses isotope labelling. It takes the atom transitions of the reactions, the isotopomer
fractions of a substrate, and the measured fragments. From these it simulates the mass isotopomer distributions by
elementary metabolite units. It fits the fluxes to the measurements from seeded starts, with 95% intervals and a
chi-square test. Select it with `create_system('carbon_13_mfa')`.

### Static systems

Solve a system of equations that has no time axis, such as a dose-response curve, and sweep its parameters. A system
under `StaticTime` has `EquationLaw`s that name their `target`, and a run solves them once and gives one frame.

```python
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.core.specs import ParameterSpec, StateSpec


class DoseElement(Element):
    def __init__(
        self,
        label: str,
        survival: Annotated[float, StateSpec()] = 0.0,
        dose: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.survival = survival
        self.dose = dose

    def _get_laws(self, s: SimpleNamespace) -> list[EquationLaw]:
        return [EquationLaw(s.survival == exp(-0.3 * s.dose - 0.03 * s.dose**2), target=s.survival)]


def dose_response(dose: float = 2.0) -> System:
    return System(elements=[DoseElement('cells', dose=dose)], time=StaticTime())


sweep = Experiment(dose_response, GridDesign(), {'dose': [0.0, 2.0]}, outputs=['cells.survival'])
rows = sweep.prepare().run().result_.rows
[round(float(row['cells.survival']), 4) for row in rows]
```

A sweep of a parameter is an `Experiment` with `GridDesign`, one static run per value, as above. The equations are
solved this way:

- The compiler solves the equations with Sage when it finds a unique explicit solution, and `problem.equations` holds
  it.
- Otherwise `RootSolver` solves them with `scipy.optimize.root` and their exact Jacobian, from the initial state as the
  guess. It raises a `SolverError` that names the equations and the largest residual when the solve does not converge.
- A target on a group gives one unknown per member, and reductions such as `total` couple the members.
- The equations of one target from every element and interaction add.
- A system without equations solves to its initial state, so its observables run on quenched states in one frame.

`PassCompiler` reports an equation with no target, or with a target that is not a state, with the code
`equation-target`. A law that does not run under static time keeps its `law-time` issue.

#### Log-sum-exp and contact clusters

Reduce over a group without overflow, and measure the clusters that points form when they touch.

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

system = create_system('random_energy_system', n_spins=12, temperature=2.0)
trajectory = Simulation(system).prepare().run(1.0).trajectory_
trajectory.values('ensemble.entropy')[0]
```

`log_sum_exp(expression, over=None)` reduces over the members of a group like `total`, as `log sum_j exp(a_j)`, without
overflow for values of any size. A system-level observable cannot reach a group from another element. So a quantity
built on it is an `EquationLaw` on a scalar element, read by an interaction that binds the group. The system
`random_energy_system` above, Derrida's random energy system, computes its entropy this way.

Quenched disorder is drawn by the factory, so a disorder average sweeps its `random_state` as an axis of `GridDesign`.
The `replicates` of an `Experiment` share one draw of the factory and vary only the dynamics.

`ClusterAnalysis(kind, radius=)` reads the positions of points, joins those closer than twice the radius, and measures
their clusters. A cluster spans along `axis` when it holds points within the radius of both faces of the space.

```python
from skcomplex.core.analysis import ClusterAnalysis

ClusterAnalysis('sizes', radius=0.6).prepare([[0, 0], [1, 0], [5, 5]]).run().sizes_.tolist()
```

### Constraints and search

Let a system search for a solution, step by step: satisfy logical constraints, optimise an objective with a population
of candidates, or anneal. Each search is an interaction on a group, so a run records its progress like any other
dynamics.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SatisfiabilityInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import CardinalitySpec, ClauseSpec, StateSpec


class BitElement(Element):
    def __init__(self, label: str, on: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0) -> None:
        super().__init__(label)
        self.on = on


constraints = [CardinalitySpec((1, 2, 3), 2, 'at-least'), ClauseSpec((-1,)), ClauseSpec((-2,))]
decide = SatisfiabilityInteraction('decide', target='bits', state='on', constraints=constraints)
bits = GroupElement('bits', element=BitElement, size=3)
system = System(elements=[bits], interactions=[decide], time=DiscreteTime())
Simulation(system).prepare().run(until=3).trajectory_.values('decide.satisfiable')[1:]
```

`SatisfiabilityInteraction` binds a group of Boolean states and reveals its constraints step by step. Each constraint is
a `ClauseSpec` or a `CardinalitySpec` over literals. A literal is a member index counted from 1, negative for its
negation. One PySAT solver keeps what it learned. It sets the members to a solution when one exists, and reports
`satisfiable` and `revealed` at every step. Above, the steps reveal that at least two of the three bits are on, then
that bit 1 is off, then that bit 2 is off. The first two constraints hold together, and the third makes the set
unsatisfiable.

`OptimizationInteraction` moves a group of candidates through pymoo, one generation per step. Each candidate is a vector
state in the box `bounds`.

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

system = create_system('evolutionary_optimisers', algorithm='particle-swarm', random_state=1)
Simulation(system, random_state=1).prepare().run(until=20).trajectory_.values('search.best')[-1] < 20
```

- The `objective` takes the array of candidates.
- The algorithm is a spec: `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, or
  `CovarianceAdaptationSpec`.
- It reports `best` and `mean` at every step. Once the algorithm stops itself, the candidates stay.
- Runs with the same seed repeat.

`SamplingInteraction(cooling=)` anneals. Its `cooling` is a function of the time symbol that multiplies one over the
intensity. So a Metropolis acceptance with the `'gaussian'` move anneals a group of candidates on a one-dimensional
`LatticeSpace` whose objective reads no neighbours, with one interaction per state. The solver of sampling takes no
other laws. So what tracks a run, such as the lowest energy so far, sits in a nested system, as in the system
`evolutionary_optimisers` with `algorithm='annealing'`.

### Control

Steer a system with a control input: drive it by a schedule, compute the optimal control for an objective, or close the
loop with a feedback controller. A control input is an adaptive parameter with bounds, annotated with
`ParameterSpec(adaptive=True, bounds=(lower, upper))`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, StateSpec


class HeaterElement(Element):
    def __init__(
        self,
        label,
        *,
        T: Annotated[float, StateSpec()] = 0.0,
        power: Annotated[float, ParameterSpec(adaptive=True, bounds=(0, 1))] = 0.0,
    ):
        super().__init__(label)
        self.T = T
        self.power = power

    def _get_laws(self, s):
        return [RateLaw(s.T, s.power - s.T)]


room = System(elements=[HeaterElement('heater')])
report = PassCompiler().prepare(System(elements=[HeaterElement('heater', power=2.0)])).report_
[issue.path for issue in report.errors]
schedule = {'heater.power': lambda t: 1.0 if t >= 1.0 else 0.0}
heated = Simulation(room, dt=0.5, schedule=schedule).prepare().run(until=3).trajectory_
heated.values('heater.power').tolist()
```

Laws read a control input like a parameter, and trajectories record it like a state. A schedule, an optimal control
solver, or a feedback controller supplies its value. Without one it keeps its initial value. The `report_` of a
compiler holds an initial value outside the bounds as an error at the parameter, and the fix of the error names the
bounds. Above, a power of 2 is outside the bounds `(0, 1)`.

#### Objectives and optimal control

Find the control that minimises or maximises a cost over a run. An objective defines a control problem, not a part of a
system, so the optimal control solvers take it as arguments.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import CollocationSolver, PontryaginSolver
from skcomplex.core.specs import ParameterSpec, StateSpec


class PlantElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        u: Annotated[float, ParameterSpec(adaptive=True, bounds=(-10, 10))] = 0.0,
    ):
        super().__init__(label)
        self.x = x
        self.u = u

    def _get_laws(self, s):
        return [RateLaw(s.x, s.u)]


plant = System(elements=[PlantElement('plant')])


def cost(s):
    return (s.plant.x**2 + s.plant.u**2) / 2


optimal = Simulation(plant, solver=PontryaginSolver(cost), dt=0.1).prepare().run(until=1.0)
round(float(optimal.trajectory_.values('plant.u')[0]), 3), round(float(optimal.stepper_.solution_.cost), 4)
transcribed = Simulation(plant, solver=CollocationSolver(cost, nodes=21), dt=0.1).prepare().run(until=1.0)
round(float(transcribed.trajectory_.values('plant.u')[0]), 3)
```

The objective has two parts:

- `objective` is the running cost.
- `terminal` is the terminal cost at the horizon, the time `until` of the run.

Each is a number, or a function of the namespace `s` of the states, parameters, and time of the system by dotted name,
such as `lambda s: s.plant.x**2`. The value of a run is the integral of the running cost over the run plus the terminal
cost. `sense` is `'min'` to minimise it or `'max'` to maximise it.

Two solvers optimise the objective over the bounded adaptive parameters of the system:

- `PontryaginSolver(objective)` builds the Hamiltonian `H = L + λ · f` of the running cost `L` and the rates `f`. It
  derives the costate equations `λ' = -∂H/∂x` and the stationarity condition `∂H/∂u = 0` symbolically, and projects
  the controls onto their bounds. Where `H` is linear in a control, it takes a bound. It solves the two-point problem by
  a forward-backward sweep.
- `CollocationSolver(objective, nodes=..., scheme=...)` transcribes the problem with the `'hermite-simpson'` or
  `'trapezoidal'` scheme, and solves it with SciPy's `minimize`, with the Jacobians derived symbolically. Collocation
  handles bang-bang controls, singular arcs, and saddle paths on which the sweep diverges.

Both solve the problem over the whole run when it starts. `simulation.stepper_.solution_` holds the solution on its
grid, its cost, and, for `PontryaginSolver`, the costates.

Every solver declares what it supports. The optimal control solvers declare continuous time, no space, rate and
equation laws, real and angle states, and elements. `PontryaginSolver` also applies boundary laws after every step of
its sweep, and `CollocationSolver` keeps the states of nonnegative domains nonnegative instead.

| Solver | Laws | Method |
| --- | --- | --- |
| `PontryaginSolver` | rate, equation, boundary | symbolic costates and a forward-backward sweep |
| `CollocationSolver` | rate, equation | direct transcription solved by SLSQP or trust-constr |

The `report_` of the compiler holds each of these as a `SolverError`:

- a system with noise, jumps, or updates run with an optimal control solver,
- a solver without an objective,
- a system under discrete time, since an objective needs static or continuous time.

```python continuation
from skcomplex.core.engine.compile import PassCompiler

[issue.code for issue in PassCompiler(solver=PontryaginSolver()).prepare(plant).report_.errors]
```

#### Feedback controllers

Close the loop with a controller of your own. The core holds no feedback controller, because a controller is an
`Interaction` you write:

- Its control input is an adaptive parameter of the plant, and it sets that parameter with a `BoundaryLaw`.
- A proportional-integral-derivative controller sets it to `kp e + ki ∫ e dt - kd y'`, from the error
  `e = setpoint - y` of a measured state. It adds a `RateLaw` for the integral of the error on an element of its own.
- A state feedback controller sets it to `-K x`.

A boundary law sets the control after every solver step. So a fixed-step solver with small steps makes the controller
close to a continuous one. The systems `pid_inverted_pendulum` and `lqr_oscillator` are built this way.

#### Linear analysis

Linearise a system near an operating point to check that it can be controlled and observed, and to design a
linear-quadratic regulator.

```python
from typing import Annotated

from skcomplex.core.analysis import LinearizationAnalysis
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.specs import ParameterSpec, StateSpec


class CartElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        v: Annotated[float, StateSpec()] = 0.0,
        force: Annotated[float, ParameterSpec(adaptive=True, bounds=(-1, 1))] = 0.0,
    ):
        super().__init__(label)
        self.x = x
        self.v = v
        self.force = force

    def _get_laws(self, s):
        return [RateLaw(s.x, s.v), RateLaw(s.v, s.force)]


linear = LinearizationAnalysis(outputs=['cart.x'], state_weights=[[1, 0], [0, 1]], control_weights=[[1]])
linear.prepare(System(elements=[CartElement('cart')])).run()
linear.controllability_, linear.observability_
linear.gain_.round(3).tolist()
```

`LinearizationAnalysis(controls, outputs, at)` finds the state-space matrices `A_`, `B_`, `C_`, and `D_` of a system
under continuous time near an operating point, from its symbolic Jacobians. The controls are named by their adaptive
parameters. Its `controllability_` and `observability_` are the ranks of the Kalman matrices. With the weights
`state_weights` and `control_weights`, its `gain_` is the gain of the linear-quadratic regulator, from SciPy's solution
of the Riccati equation.

The catalog systems with optimal controls are `optimal_harvesting` in resources, `optimal_consumption` in
macroeconomics, `sir_vaccination` in epidemiology, and `optimal_chemotherapy` and `optimal_immunotherapy` in oncology.

### Learning by gradient steps

Fit the parameters of a system to data by gradient descent, as a neural network is trained. A system that learns is a
dynamical system in two times:

- The inner time steps a system through its inputs.
- The outer time steps through training. The trained parameters of the inner system are its states, and a gradient step
  on a loss of the runs of the inner system is its law.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import TrainingInteraction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import DatasetSpec, ObservableSpec, ParameterSpec, StateSpec


class DecayElement(Element):
    def __init__(
        self,
        label,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        a: Annotated[float, ParameterSpec(adaptive=True)] = 0.5,
        observed: Annotated[float, ParameterSpec()] = 0.0,
    ):
        super().__init__(label)
        self.x = x
        self.a = a
        self.observed = observed

    def _get_laws(self, s):
        return [UpdateLaw(s.x, s.a * s.x)]


def squared_error(s):
    return (s.unit.x - s.unit.observed) ** 2


series = 0.8 ** np.arange(1, 11)
system = System('decay', elements=[DecayElement('unit')], time=DiscreteTime())
training = TrainingInteraction(
    'fit',
    system='decay',
    trained=['unit.a'],
    loss=ObservableSpec('error', squared_error),
    data=DatasetSpec({'unit.observed': series[np.newaxis]}),
    learning_rate=0.05,
    batch_size=1,
)
fitting = System('fitting', systems=[system], interactions=[training], time=DiscreteTime())
trajectory = Simulation(fitting, random_state=0).prepare().run(until=300).trajectory_
round(float(trajectory.values('decay/unit.a')[-1]), 3), bool(trajectory.values('fit/loss')[-1] < 1e-6)
```

`TrainingInteraction`, of `skcomplex.core.components.interactions`, writes this with the existing parts:

- The inner system is a nested system, such as `decay` above.
- Its trained parameters are declared `ParameterSpec(adaptive=True)`.
- The interaction binds the inner system through its role `system`.
- The outer system runs under `DiscreteTime()`, one step per epoch.

`TrainingInteraction` returns one `GradientStepLaw`, of `skcomplex.core.components.laws`, which differs from
`GradientInteraction`:

- `GradientInteraction` moves the states of its target down the gradient of a potential, in the system's own time, and
  lowers to rate laws.
- `GradientStepLaw` takes optimizer steps on the adaptive parameters of a nested system once per epoch. It follows the
  gradient of a loss over unrolled runs of that system, computed by automatic differentiation with JAX.

The parts of training work as follows:

- **The inner system.** It is an ordinary system, a discrete map or ordinary differential equations written as symbolic
  laws, so it runs, is analysed, and is visualised alone, before or after training. The trained system of a frame is
  the nested system with the recorded values set through `set_params`. A system with noise, jumps, rules, or laws given
  as Python callables reports `learning-not-differentiable`.
- **The data.** `DatasetSpec(values, weights=None, times=None)`, of `skcomplex.core.specs`, maps dotted parameter paths
  of the inner system, its inputs and targets, to arrays of shape `(sequences, steps)` followed by the shape of the
  parameter. At inner step `k`, every parameter takes its value at step `k` before the laws of the step are evaluated,
  as a schedule sets a parameter, and the loss reads the states that the step gives. Every sequence starts from the
  initial states of the system. `weights` masks steps out of the loss. It is one array for every observable, or a
  mapping from the label of the loss or of a metric to its own weights. `times` gives the inner times of an ODE system,
  `1, 2, ...` by default.
- **The loss and the metrics.** `loss` and `metrics` are `ObservableSpec`s over the symbols of the inner system. The
  loss of a batch is the weighted mean of the observable over its sequences and steps. The trajectory records the
  trained parameters under their inner paths, such as `decay/unit.a`, the mean batch loss of every epoch as
  `<interaction>/loss`, and every metric over the whole dataset as `<interaction>/<metric>`. The same observables given
  to a `Simulation` of the trained system record the loss in a forward run.
- **The optimizer.** `optimizer` is one of `OPTIMIZERS`, `'sgd'`, `'momentum'`, `'rmsprop'`, and `'adam'`, run by
  Optax. Its hyperparameters, `learning_rate`, `momentum`, `beta1`, and `beta2`, the decay of `'rmsprop'`, are
  parameters of the interaction. So a schedule such as `{'fit.learning_rate': ...}` changes them at the next frame.
  `batch_size` sets the number of sequences of a batch, shuffled by the random stream of the simulation, so a seed
  repeats a run.
- **Truncation.** `truncation` cuts backpropagation through time after that many inner steps, and `None` follows the
  gradient through the whole run. A truncation of 1 holds the previous inner states as constant inputs, Elman's
  original rule.
- **The solver.** `GradientDescentSolver`, of `skcomplex.core.engine.solvers`, is the default solver of a system that
  trains. It unrolls a discrete map with `jax.lax.scan` and an ODE with diffrax, computes in double precision by
  default, and keeps the state of the optimizer between frames. The inner system always compiles with `JaxBackend`, so a
  backend passed to the simulation is ignored.
- **Flat functions.** `heaviside`, `sign`, `floor`, and `ceil` have a zero gradient almost everywhere. So a loss that
  reads a parameter only through them does not train it, without any error.
- **Local rules.** Local learning rules, such as Hebb's and Oja's, change a weight from its own activity. So they are
  update or rate laws on weight states, as in the `hopfield` system, and need no training interaction.

The system `elman_network` trains Elman's simple recurrent network on the temporal XOR stream. Its visual draws the loss
and the metrics as the first view, and the trained matrices as heat maps through `MatrixVisual`, chosen by
`VisualSpec('matrix')`.

#### Shaped parameters and coupling tensors

Give an interaction a weight matrix, or a tensor of higher order, whose shape follows the groups it binds.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import contraction, count, total
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class NodeElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


class DriveInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        W: Annotated[object, ParameterSpec(shape=('target', 'source'))] = 0.0,
    ):
        super().__init__(label)
        self.target = target
        self.source = source
        self.W = W

    def _get_laws(self, s):
        return [RateLaw(s.target.x, -s.target.x + total(s.source.x, weights=s.W))]


W = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 2.0]])
system = System(
    elements=[
        GroupElement('cells', element=NodeElement, size=3),
        GroupElement('inputs', element=NodeElement, size=2, values={'x': [1.0, 2.0]}),
    ],
    interactions=[DriveInteraction('drive', target='cells', source='inputs', W=W)],
)
simulation = Simulation(system, dt=0.1).prepare()
simulation.run(until=20).trajectory_.values('cells.x')[-1].round(3).tolist()
simulation.update_parameters({'drive.W': 2 * W})
simulation.run(until=40).trajectory_.values('cells.x')[-1].round(3).tolist()
```

`ParameterSpec(shape=('target', 'source'))` declares a parameter of an interaction whose shape is the sizes of the
groups its roles bind. Here it is a weight matrix with one row per target member and one column per source member. It
behaves like any parameter:

- It holds a block of the parameter vector in row-major order.
- Trajectories and `Simulation.update_parameters` read and set it with its shape, and a schedule or an experiment
  changes it.
- With `adaptive=True`, a training interaction trains it.
- The weights of `total` and `mean`, and of `CouplingInteraction(mode='group')`, read it.

A shaped parameter whose role binds a single element reports the issue `parameter-shape`, with the fix of binding a
group.

A role repeated `k` times, `shape=('target',) * k`, gives a coupling tensor of order `k` among the members of a group.
`contraction(T, a, b)`, of `skcomplex.core.components.laws.utils`, returns `sum_jk T_ijk a_j b_k` for every member `i`,
with one expression per index after the first. Matrices keep `total(x, weights=W)`.

```python continuation
class TripletInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        T: Annotated[object, ParameterSpec(shape=('target',) * 3)] = 0.0,
    ):
        super().__init__(label)
        self.target = target
        self.T = T

    def _get_laws(self, s):
        x = s.target.x
        return [RateLaw(x, x * (1 - x) + x * contraction(s.T, x, x) / count(over=s.target))]


species = GroupElement('species', element=NodeElement, size=4, values={'x': 0.1})
triplets = TripletInteraction('triplets', target='species', T=-np.ones((4, 4, 4)))
community = System(elements=[species], interactions=[triplets])
trajectory = Simulation(community, dt=0.1).prepare().run(until=50).trajectory_
trajectory.values('species.x')[-1].round(3).tolist()
```

A tensor of order `k` over `N` members holds `N^k` entries, a million for order 3 over 100 members. The system
`higher_order_random_lotka_volterra` builds the random community of Sidhom and Galla (2024) on such a tensor.

## Simulate

Run a system frame by frame, choose its solver and backend, hand parts of it to external simulators, and couple
simulators together.

### Simulating

Run any system, ready-made or your own, for as long as you want, change its parameters on the way, and continue a run
where it stopped. `Simulation` compiles the system, picks the solver of its form, and advances it in frames of `dt`.

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

system = create_system('lotka_volterra')
trajectory = Simulation(system, dt=0.1).prepare().run(until=10).trajectory_
trajectory.values('prey.n')[-1].round(3)
```

`prepare` validates the system, compiles it, and chooses the solver. `run(until)` runs from time 0 to `until`, and
`trajectory_` holds the recorded frames. States and parameters are named by their dotted path, such as `prey.n` and
`predation.a`.

The default ODE solver is `ScipyODESolver` with SciPy's `LSODA`. It starts with Adams methods and switches to BDF
methods while the system is stiff, so stiff systems also run with the default solver.

#### Changing a run

Change a parameter between frames with `update_parameters`, and continue a finished run with `extend`.

```python
from sklearn.base import clone

from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

simulation = Simulation(create_system('lotka_volterra'), dt=0.1).prepare().run(until=10)
simulation.update_parameters({'predation.a': 0.0})
len(simulation.extend(until=20).trajectory_.times)
clone(simulation).prepare().run(until=5).time_
```

- `update_parameters({'predation.a': 0.05})` changes parameters before the next frame. Values set between runs stay in
  force for the later runs of the same simulation, and `clone(simulation).prepare().run(until)` starts afresh.
- `extend(until)` continues a finished run from its last frame, with the same stepper, clock, and observers, and
  appends the new frames to the trajectory. `run` starts again from time 0.

#### Run settings

Set how a run starts and evolves with the arguments of `Simulation`.

```python
from skcomplex.core.engine.runtime import RealTimeClock, Simulation
from skcomplex.systems import create_system

simulation = Simulation(
    create_system('lotka_volterra'),
    dt=0.1,
    schedule={'predation.a': lambda t: 0.4 + 0.001 * t},
    initial_state={'prey.n': 20},
    clock=RealTimeClock(speed=1000.0),
).prepare()
simulation.run(until=10).trajectory_.values('prey.n')[0]
simulation.start(until=20)
simulation.wait()
simulation.time_
```

- `dt` defaults to 1 for discrete maps and agents. For other forms it defaults to 0.01, raised to the smallest of 0.02,
  0.05, 0.1, and so on, in steps of 1, 2, and 5 times a power of ten, that keeps the run at most 1000 frames long. So
  `run(until=1000)` takes frames of 1.
- `until` sets the end time that `run` uses when it is given none.
- `clock=RealTimeClock(speed)` paces frames so that simulated time advances `speed` times faster than wall time.
- `start(until)` runs in a background thread, and `pause`, `resume`, `stop`, and `wait` control it.
- `random_state` makes stochastic runs reproducible. A fixed-step solver splits every frame into
  `step_count(t, until, dt)` equal steps on every backend, so the same `random_state` gives the same path on the NumPy
  and Numba backends.
- `schedule={'predation.a': lambda t: 0.4 + 0.001 * t}` sets parameters from functions of time at every frame, such as
  a slow ramp of a control parameter.
- `initial_state={'prey.n': 20}` replaces initial values of the elements for one simulation.

### Backends and speed

Run large systems fast by choosing the backend that turns their laws into numeric functions. Compiling a system
produces these functions with a backend.

```python
from skcomplex.core.engine.backends import NumbaBackend
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.systems import create_system

gas = create_system('ideal_gas', n_molecules=50_000, random_state=0)
simulation = Simulation(gas, solver=RK4Solver(dt=0.01), backend=NumbaBackend(), dt=0.01).prepare()
simulation.run(until=0.5).problem_.backend
```

The library has three backends:

- `NumpyBackend` evaluates every law with NumPy arrays.
- `NumbaBackend` generates one compiled function per form. It adds compiled cell lists for pair interactions and
  compiled loops for `RK4Solver`, `EulerMaruyamaSolver`, `GillespieSolver`, and `IterationSolver`.
- `JaxBackend` writes discrete maps and ordinary differential equations as pure JAX functions, which JAX traces,
  compiles, and differentiates. It is the backend of the inner system of a system that trains, and a forward backend
  that agrees with NumPy within 1e-10 in double precision.

Without a backend, `Simulation` picks Numba when the groups of a system hold at least 1000 members in total. Compiling
takes a second or two, which small systems do not win back. `problem_.backend` names the backend a run used.

Compiled kernels are cached on disk, under `~/.cache/skcomplex` or the `cache_dir` of `NumbaBackend(cache_dir=...)`.
So running the same system again skips compilation.

#### Compiled functions in laws

Call a fast function of your own inside a law. A law value can be a function called once per member, with
`CallSpec(function, arguments)`.

```python
from typing import Annotated

import numba

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import CallSpec, ParameterSpec, RoleSpec, StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


@numba.njit
def saturating(t, n, r):
    return r * n / (1 + n)


class SaturationInteraction(Interaction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], r: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s):
        return [RateLaw(s.target.n, CallSpec(saturating, (s.target.n, s.r)))]


herds = System(
    elements=[GroupElement('herds', element=PopulationElement, size=2000, values={'n': 1.0})],
    interactions=[SaturationInteraction('growth', target='herds')],
)
round(float(Simulation(herds, dt=0.1).prepare().run(until=1).trajectory_.values('herds.n')[-1, 0]), 4)
```

A Numba-compiled function runs inside the compiled kernels, and the NumPy backend calls it from Python. The group above
holds 2000 members, so the run picks Numba by itself.

A plain Python callable of the values, as in `RateLaw(s.target.n, lambda t, values: ...)`, also works with the Numba
backend. It runs in Python, though, and turns off the compiled stepper loops for that kind of law.

### External backends

Hand a whole system to a third-party simulator, and keep the frames, trajectories, live parameters, and Rerun
visualization of the library. `LammpsSolver` runs particle systems in LAMMPS, which integrates them with velocity
Verlet.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.components.elements import GroupElement, PointElement
from skcomplex.core.components.interactions import PairCouplingInteraction, RateInteraction
from skcomplex.core.components.laws.utils import derivative
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.solvers import LammpsSolver
from skcomplex.core.specs import ParameterSpec


class LennardJonesInteraction(PairCouplingInteraction):
    def __init__(
        self,
        label,
        first,
        second,
        cutoff=1.0,
        position='x',
        *,
        epsilon: Annotated[float, ParameterSpec()] = 1.0,
        sigma: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label, first, second, cutoff=cutoff, position=position, potential=self._get_energy)
        self.epsilon = epsilon
        self.sigma = sigma

    @staticmethod
    def _get_energy(s, r):
        return 4 * s.epsilon * ((s.sigma / r) ** 12 - (s.sigma / r) ** 6)


class DragInteraction(RateInteraction):
    def __init__(self, label, *, target, gamma: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, target=target, state='x', rate=self._get_rate)
        self.gamma = gamma

    @staticmethod
    def _get_rate(s):
        return -s.gamma * s.target.x.apply_map(lambda item: derivative(item, 1))


def get_pair_coefficients(value):
    return f'{float(value("epsilon")[0])} {float(value("sigma")[0])}'


def get_drag_fix(fix):
    return f'fix {fix.name} {fix.group} viscous {float(fix.value("gamma")[0])}'


side = np.arange(10) * 1.2
grid = np.stack(np.meshgrid(side, side), axis=-1).reshape(-1, 2)
atoms = System(
    elements=[GroupElement('atoms', element=PointElement, size=100, values={'x': grid})],
    interactions=[
        LennardJonesInteraction('lj', 'atoms', 'atoms', cutoff=2.5),
        DragInteraction('air', target='atoms', gamma=1.0),
    ],
    space=ContinuousSpace(n_dim=2),
)
pairs = {LennardJonesInteraction: ('lj/cut', get_pair_coefficients)}
fixes = {DragInteraction: get_drag_fix}
solver = LammpsSolver(dt=0.005, pairs=pairs, fixes=fixes)
trajectory = Simulation(atoms, solver=solver, dt=0.05, random_state=0).prepare().run(until=1).trajectory_
trajectory.values('atoms.x').shape
```

Two maps tell the solver how to write your interactions in LAMMPS:

- `pairs` maps pair potential classes to native LAMMPS pair styles, with a function that writes their coefficients.
- `fixes` maps interaction classes to native fixes, with a function that writes the fix.

You write both maps for your own classes, as above, and the prebuilt particle systems come with theirs. A function in
`fixes` reads a seed from the random state of the simulation with `fix.seed()`, for a stochastic fix such as
`fix langevin`.

Before the first frame the backend checks the system. It raises one error that lists every element, interaction, and
law it cannot run. The library translates a system to LAMMPS as follows:

- Elements and groups with a vector position of order 2, such as `PointElement`, become one atom type per element or
  group. The velocities of the atoms are the first derivatives of the positions. Their mass is the parameter that the
  `divisor` of their interactions names, or 1 without one.
- Two or three dimensions become `dimension 2` with `fix enforce2d`, or `dimension 3`.
- `ContinuousSpace(extent, boundary='reflecting')` becomes `fix wall/reflect` at `0` and the extent.
- `ContinuousSpace(extent, boundary='periodic')` becomes periodic boundaries of the extent, shared by every group.
- A class in `pairs`, such as `LennardJonesInteraction`, becomes its native pair style, such as `pair_style lj/cut`.
- Any other `PairCouplingInteraction` with a potential becomes `pair_style lepton`, with the potential printed as a
  Lepton expression.
- A class in `fixes` becomes its native fix, such as `fix addforce`, `fix viscous`, or `fix langevin`.
- Any other interaction that only changes the derivatives of the positions becomes `fix external`, which the library
  evaluates at every step.

### Co-simulation

Run parts of a system that need different solvers side by side, such as farming agents next to the differential
equation of the land they harvest. A system whose laws do not fit one form is split into partitions at subsystem
boundaries, and each partition runs with its own solver.

```python
from skcomplex.core.engine.compile import compile_system
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

farming = create_system('harvesting', farmers=10, effort=0.05)
[(partition.name, partition.problem.form.value) for partition in compile_system(farming).partitions]
trajectory = Simulation(farming, random_state=0).prepare().run(until=50).trajectory_
round(trajectory.values('land/stock.n')[-1]), trajectory.agents('village/farmers', 50).values['wealth'].size
trajectory.values('village/farmers_count.value')[-1]
```

Every subsystem whose laws fit one form becomes a partition. A law joins the partition that owns its target, so an
interaction of the parent system may contribute to several partitions. A subsystem whose own elements mix law kinds
raises `IncompatibleLawsError`, with its path and the suggestion to split it.

The partitions exchange values in these ways:

- Every partition holds the states of the other partitions as inputs. They keep their values from the last
  synchronisation point while it advances.
- A law of one partition reads a reduction of a group of another through a role bound to the group, such as
  `count(s.farmers)` or `total(s.farmers.wealth)`. The co-simulation computes every reduction of agents at each
  synchronisation point and holds it until the next, in the state `value` of an element next to the group, such as
  `village/farmers_count.value`. A reduction of an element group reads the states of the other partition, which keep
  their values from the last synchronisation point too.
- Agent rules read the states of other partitions through `system.inputs[name]`, such as
  `system.inputs['land/stock.n']`.

The `CoSimulationSolver` advances the partitions and controls the exchange:

- It advances every partition with its own solver, chosen per partition through `solvers={'land': ScipyODESolver()}`.
- It exchanges inputs every `sync_dt`. This defaults to the frame duration and must divide it, and it must be a whole
  number of steps for discrete and agent partitions.
- The Gauss-Seidel scheme, the default, advances partitions in system order, each reading the values already advanced
  by those before it. `scheme='jacobi'` advances all of them from the values at the start of the interval.
- Splitting adds an error of first order in `sync_dt`. Halving `sync_dt` roughly halves it, at the cost of more
  exchanges. `CoSimulationSolver(split=True)` splits a system that fits one form, which shows the error against the
  joint solution.

## Observe

Record the quantities you care about at every frame, check that conserved ones stay constant, and watch a run live.

### Observables

Record the quantities you care about at every frame, such as the order of a group of oscillators, beside the states of
a run. An `ObservableSpec` is a label with an expression or a group reduction, which a run computes at every frame,
records in trajectories under the path of the system, and logs to Rerun.

```python
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import I, exp, pi

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.specs import ObservableSpec, ParameterSpec, StateSpec


class OscillatorElement(Element):
    def __init__(
        self,
        label,
        *,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        omega: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.theta = theta
        self.omega = omega

    def _get_laws(self, s):
        return [RateLaw(s.theta, s.omega)]


def phase_vector(member):
    return exp(I * member.theta)


def order_parameter(s):
    return abs(s.z)


values = {'theta': np.linspace(0, 1, 100), 'omega': np.linspace(0.9, 1.1, 100)}
oscillators = System(elements=[GroupElement('phases', element=OscillatorElement, size=100, values=values)])
observables = [ObservableSpec('z', phase_vector, group='phases'), ObservableSpec('order', order_parameter)]
order = Simulation(oscillators, dt=1.0, observables=observables).prepare().run(until=50).trajectory_.values('order')
round(float(order[0]), 3), round(float(order[-1]), 3)
```

An observable is a frozen record of one of two kinds:

- **An expression of the system namespace.** Earlier observables appear in it by label, as `s.z` in `order_parameter`.
- **A reduction over the members of a group.** It reduces a variable, or an expression of one member such as
  `phase_vector`, by `count`, `sum`, `mean`, `min`, `max`, `var`, or a function.

Expressions and reductions given as functions must be module-level functions, so that systems pickle. A lambda raises
an error.

Above, the oscillators start close together and drift apart at their own frequencies, so the order falls from near 1.

#### Laws that read observables

Let a law react to a macroscopic quantity, such as the mean field of all oscillators. A law reads an observable through
a role bound to the path of the observable, as `s.<role>.value`.

```python continuation
from sage.all__sagemath_symbolics import cos, sin

from skcomplex.core.components.interactions import Interaction
from skcomplex.core.specs import RoleSpec


class MeanFieldInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec()],
        X: Annotated[str, RoleSpec()],
        Y: Annotated[str, RoleSpec()],
        K: Annotated[float, ParameterSpec()] = 1.0,
    ):
        super().__init__(label)
        self.target = target
        self.X = X
        self.Y = Y
        self.K = K

    def _get_laws(self, s):
        return [RateLaw(s.target.theta, s.K * (s.Y.value * cos(s.target.theta) - s.X.value * sin(s.target.theta)))]


def cosine(member):
    return cos(member.theta)


def sine(member):
    return sin(member.theta)


values = {'theta': np.linspace(0, 6, 100), 'omega': np.linspace(0.9, 1.1, 100)}
coupled = System(
    elements=[GroupElement('phases', element=OscillatorElement, size=100, values=values)],
    interactions=[MeanFieldInteraction('coupling', target='phases', X='X', Y='Y', K=2.0)],
)
observables = [ObservableSpec('X', cosine, group='phases'), ObservableSpec('Y', sine, group='phases')]
synchronised = Simulation(coupled, dt=1.0, observables=observables).prepare().run(until=20).trajectory_
round(float(synchronised.values('X')[-1]), 2)
```

When the law reads the value depends on where the observable and the law live:

- A symbolic observable is an expression, or a reduction with a string `reduce` over an element group. Laws of its own
  partition read it at the current state. The compiled problem evaluates it before the law, so the value the law reads
  equals the recorded value at every frame.
- A law of another partition reads the observable as held at the last synchronisation point. This is how agents and
  equations exchange macroscopic quantities in a co-simulation.
- An observable with a Python function as `reduce`, or one over agents, is read only from another partition. Reading it
  in its own partition raises an error.

### Visualizing with Rerun

Watch a run live in the [Rerun](https://rerun.io) viewer, or save it to a file to open later. `RerunObserver` streams
frames to a spawned viewer, a running viewer, or a `.rrd` file.

```python
import tempfile
from pathlib import Path

from skcomplex.core.components.elements import PointElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.engine.viz import RerunObserver

molecule = System(
    elements=[PointElement('a', x=[0.0, 0.0, 0.0]), PointElement('b', x=[1.5, 0.0, 0.0], x_d1=[0.0, 0.5, 0.0])],
    interactions=[
        RateInteraction('pull', target='a', sources=['b'], rate=lambda s: 4.0 * (s.sources[0].x - s.target.x)),
        RateInteraction('push', target='b', sources=['a'], rate=lambda s: 4.0 * (s.sources[0].x - s.target.x)),
    ],
    space=ContinuousSpace(n_dim=3),
)
path = Path(tempfile.mkdtemp()) / 'molecule.rrd'
Simulation(molecule, dt=0.02, observers=[RerunObserver('molecule', save_path=path)]).prepare().run(until=5)
```

Entity paths follow the system tree, each element type chooses its visual, and parameters appear under `parameters/`.
The default visuals are these:

- Particles appear as 3D points, and populations as time series.
- Agents on grids and in continuous spaces appear as points colored by a state, and agents on networks as graph nodes.
- Reporters appear under `reporters/`.
- A group is logged as one batch per state. `PointsVisual(position='x', color_by='x_d1')` colors its points by speed.
- Points and agents in a continuous space without a given `radius` take a share of the extent of the view. A group of
  at most ten points draws the trail of its recent positions, which `trail=0` turns off.
- A one-dimensional lattice also logs a space-time image, one row per logged frame with the oldest at the top, holding
  at most `ImageVisual(history=...)` rows.
- A complex state of a lattice or a mesh appears as two images. Its modulus goes under `<path>/<state>/modulus`, colored
  over the range, and its phase under `<path>/<state>/phase`, colored by a cyclic palette over `(-π, π]`. A complex
  scalar appears as the series `<path>/<state>/real` and `<path>/<state>/imag`.

An element or a group sets its own visual with its `visual` argument, such as
`GroupElement(..., visual=VisualSpec('points', {...}))`. A `VisualSpec(kind, options)` names a visual of the
visualization layer, such as `'points'`, `'image'`, or `'agents'`. `RerunObserver` builds it with the path of the
element and its recording.

Every visual subclasses `BaseVisual`, and implements the hook `_log` for elements or `_log_agents` for agents. Its
settings `path` and `recording` name the element and the recording it logs to. `visual.prepare(problem)` finds the
element in a compiled problem and sets its views `views_`, and `visual.run(frame)` logs the element at a frame.

#### Layouts and blueprints

Arrange the views of the viewer with a Rerun blueprint, the default one or your own. Every visual declares the Rerun
views that show what it logs, as its fitted attribute `views_`.

```python continuation
import rerun.blueprint as rrb

from skcomplex.systems import create_system

predation = create_system('lotka_volterra')
observer = RerunObserver('lotka_volterra', save_path=path.with_name('lotka_volterra.rrd'), blueprint='default')
Simulation(predation, dt=0.1, observers=[observer]).prepare().run(until=20)
type(observer.views_[0]).__name__
sorted(file.name for file in path.parent.glob('lotka_volterra.*'))
custom = rrb.Blueprint(rrb.TimeSeriesView(origin='/prey', name='prey'), collapse_panels=True)
observer = RerunObserver('lotka_volterra', save_path=path.with_name('prey.rrd'), blueprint=custom)
Simulation(predation, dt=0.1, observers=[observer]).prepare().run(until=20)
path.with_name('prey.rbl').exists()
```

The views follow what is logged:

- Scalars go in time series views.
- Points go in 2D or 3D spatial views, by their dimension.
- Lattices and fields go in one 2D view per state, after the 2D views of their space-time images in one dimension.
- Meshes go in 3D views.
- Agents go in a 2D view, or a graph view on a network. Agents without a space get no spatial view.

`RerunObserver(blueprint='default')` arranges these views into a Rerun blueprint:

- Spatial views come first and side by side, with time series views below them.
- Scalar elements of one type share a time series view with at most six series.
- Parameters are left out unless `parameters=True`.
- The side panels are collapsed, the top panel is hidden, and the time panel follows the latest frame.

`primary` names the view to put first. The prepared observer keeps the views in this order as `views_`, so its first
view is the one that shows the signature behaviour of the system. `RerunObserver` sends the default blueprint or a
blueprint of your own, `blueprint_`, and writes it next to a saved recording as a `.rbl` file.

Open both files with `rerun lotka_volterra.rrd lotka_volterra.rbl` to see the layout.

## Analyse

Derive the equations of a system, and measure its runs, its data, and its structure.

### Analysing symbolically

Read the equations a system stands for, and solve them exactly where a closed form exists. The analysis works on the
symbolic laws, so it needs no run.

```python
from skcomplex.core.analysis import SymbolicAnalysis
from skcomplex.core.components.elements import PointElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.components.system import System

oscillator = System(
    elements=[PointElement('p', x=1.0)],
    interactions=[RateInteraction('trap', target='p', rate=lambda s: -4.0 * s.target.x)],
    space=ContinuousSpace(n_dim=1),
)
SymbolicAnalysis().prepare(oscillator).run().equations_['p.x[0]']
SymbolicAnalysis('solve', use_initial_state=True).prepare(oscillator).run().solutions_['p.x[0]']
```

- `SymbolicAnalysis('equations')`, the default kind, returns the equations of a system as Sage relations.
- `SymbolicAnalysis('solve')` solves an ODE system exactly where Maxima can. It uses `desolve` for one equation and
  `desolve_system` for linear systems with constant coefficients. With `use_initial_state=True`, the initial state of
  the system gives the initial conditions at `t = 0`. Without it, the solution keeps its integration constants.

### Analysing dynamics and data

Measure what a system does: its stability and chaos, its fluctuations and information, its patterns, agents, and
networks. `skcomplex.core.analysis` measures complex dynamics in the same way for every field. Every measure takes a
system, or plain arrays from trajectories or measured data.

**Stability and chaos.** Find the fixed points of a system, how chaotic it is, and how its long-term behaviour changes
with a parameter.

```python
from skcomplex.core.analysis import LyapunovAnalysis, StabilityAnalysis
from skcomplex.core.experiments import OrbitDiagramExperiment
from skcomplex.systems import create_system, describe

StabilityAnalysis().prepare(create_system('lotka_volterra')).run().classifications_
LyapunovAnalysis(until=200).prepare(create_system('lorenz')).run().exponents_.round(2).tolist()
logistic_map = describe('logistic_map').factory
diagram = OrbitDiagramExperiment(logistic_map, 'map.r', [2.9, 3.2, 3.5], 'map.x', transient=500, keep=32, dt=1)
diagram.prepare().run().periods_.tolist()
```

- `StabilityAnalysis` finds the fixed points, the Jacobian, the stability, the nullclines, and the divergence of the
  symbolic vector field of a flow or a map.
- `LyapunovAnalysis` integrates the tangent dynamics, compiled with Numba.
- The experiments `OrbitDiagramExperiment` and `BasinStabilityExperiment` run a system over parameters or initial
  conditions.

**Sections.** Reduce a flow to a map by sampling it at chosen moments.

```python
import math

import numpy as np

from skcomplex.core.analysis import SectionAnalysis
from skcomplex.systems import create_system

section = SectionAnalysis('stroboscopic', period=2 * math.pi / 1.2, n_points=200)
section = section.prepare(create_system('duffing')).run()
len(np.unique(np.round(section.states_, 4), axis=0))
```

`SectionAnalysis('poincare')` finds the states where a state crosses a value in a direction, located like an event.
`SectionAnalysis('stroboscopic')` samples a periodically forced system once per period.

**Distributions and events.** Describe heavy tails, and cut series into avalanches or bursts.

```python
import numpy as np

from skcomplex.core.analysis import DistributionAnalysis, EventAnalysis, FitAnalysis

samples = (1 - np.random.default_rng(0).random(5000)) ** (-1 / 1.5)
fit = FitAnalysis('power', bootstrap=20, random_state=0).prepare(samples).run()
gini = DistributionAnalysis('gini').prepare(samples).run().gini_
round(fit.exponent_, 2), fit.p_value_, round(gini, 2)
EventAnalysis().prepare([0, 2, 3, 0, 0, 1, 0]).run().sizes_.tolist()
```

- `DistributionAnalysis` describes heavy tails by its `kind`: a `'histogram'` with logarithmic bins, the `'ccdf'`, the
  Hill `'tail-index'`, and the `'gini'` coefficient.
- `EventAnalysis` turns series into avalanches or bursts, with sizes, durations, and waiting times.
- `FitAnalysis(law='power')` adapts the `powerlaw` library. It chooses the lower cutoff by the Kolmogorov-Smirnov
  criterion of Clauset, Shalizi, and Newman, and accepts an upper cutoff. It uses the exact discrete likelihood for
  integers, and gives a bootstrap p-value. It compares the power law with the exponential, log-normal, stretched
  exponential, and truncated power-law tails of `compare_with`.

**Time series and information.** Measure the memory, diffusion, entropy, and information flow of series.

```python
import numpy as np

from skcomplex.core.analysis import EntropyAnalysis, InformationAnalysis, ScalingAnalysis

noise = np.random.default_rng(1).standard_normal(20_000)
ScalingAnalysis().prepare(np.column_stack([noise, np.cumsum(noise)])).run().exponent_.round(2).tolist()
periodic = [0, 0, 1, 1] * 200
[EntropyAnalysis(kind, dimension=4).prepare(periodic).run().entropy_.round(3).tolist() for kind in ('rate', 'excess')]
driver = np.random.default_rng(2).integers(0, 2, 5000)
transfer = InformationAnalysis('transfer').prepare(np.column_stack([driver, np.roll(driver, 1)])).run()
round(transfer.information_[0, 1], 2)
```

- `CorrelationAnalysis`, `SpectrumAnalysis`, `DisplacementAnalysis`, `ScalingAnalysis`, and `RollingAnalysis` describe
  memory and diffusion.
- The mean squared displacement of `DisplacementAnalysis` comes from an FFT over all lags of unwrapped positions. Its
  exponent and diffusion coefficient are fitted over a chosen range of lags.
- The fluctuation analyses come from `nolds`. Detrended fluctuation analysis of any order gives `H + 1` for an
  integrated series such as a random walk, as above. The rescaled range is the other kind.
- `RollingAnalysis` gives the variance, the lag-one autocorrelation, or the skewness in rolling windows, with its trend.
  A rising trend of these is an early warning of tipping.
- `InformationAnalysis` measures structure and information flow by its `kind`, such as `'mutual'` and `'transfer'`, as
  in Lindgren's information theory of complex systems.
- `EntropyAnalysis` gives the Shannon, block, rate, excess, permutation, and Lempel-Ziv entropies by its `kind`. Its
  setting `symbols` turns real series into symbols first: bins of equal width, equally populated bins, or ordinal
  patterns. Ordinal patterns are coded by the lexicographic rank of their permutation with `ordpy`, with a delay and of
  any length. The Lempel-Ziv parsing comes from `antropy`.

**Space and networks.** Measure clusters, fractals, and graphs.

```python
import networkx as nx
import numpy as np

from skcomplex.core.analysis import ClusterAnalysis, NetworkAnalysis, ScalingAnalysis

clusters = ClusterAnalysis('labels').prepare(np.random.default_rng(3).random((100, 100)) < 0.6).run()
clusters.spanning_, clusters.sizes_.max()
rows, columns = np.indices((512, 512))
round(ScalingAnalysis('box-counting').prepare((rows & columns) == columns).run().exponent_, 2)
NetworkAnalysis('summary').prepare(nx.watts_strogatz_graph(200, 6, 0.1, seed=0)).run().diameter_
```

`ClusterAnalysis` labels the clusters of a grid, with periodic edges. `CorrelationAnalysis` gives radially averaged
correlations, and `ScalingAnalysis` measures fractals, such as the Sierpinski triangle above. `NetworkAnalysis` measures
the degrees, the summary, and the centralities of NetworkX graphs and of the networks of trajectories.

#### Kinds of data and signatures

Find which analyses apply to a trajectory or an array, and get a default picture of each kind of data it holds. Every
analysis consumes one kind of data, never a system or a field of science: a series, a jump path, a field, agent
snapshots, a network, or an ensemble of runs.

```python
import numpy as np

from skcomplex.core.analysis import DiscoveryAnalysis, SpectrumAnalysis
from skcomplex.systems import create_simulation

spins = create_simulation('ising', size=32).prepare().run(until=20).trajectory_
DiscoveryAnalysis().prepare(spins).kinds_
SpectrumAnalysis('structure-factor').prepare(spins).run().wavenumbers_[:3].round(3).tolist()
discovery = DiscoveryAnalysis().prepare(np.zeros((16, 16))).run()
discovery.readings_['series'], len(discovery.analyses_['field']) > 0
```

The same `SpectrumAnalysis` serves an Ising lattice and a reaction-diffusion field. The same
`EventAnalysis('excursions')` serves a sandpile and an earthquake system. Every analysis takes plain arrays or graphs,
so it works on measured data as well.

An analysis prepared on a trajectory extracts its kind of data itself:

- named series of states, reporters, or observables,
- the jump path of a discrete state,
- the frames of a lattice,
- the agents of a group, with their positions unwrapped across periodic sides,
- the network of every frame.

An ensemble of runs is one row per run and one column per common time, as `ExperimentResult.ensemble` returns it.

`DiscoveryAnalysis` prepares on a trajectory, a system, or an array, and keeps the kinds of data the object holds in
`kinds_`. Its run lists in `analyses_` the analyses of the library that apply under each kind, and in `readings_` how
they read it.

`SignatureAnalysis` computes one default analysis per kind of data of a trajectory, with its curves, axis labels, title,
and measured values. The gallery draws it for every example that names no analysis of its own.

```python
from skcomplex.core.analysis import SignatureAnalysis
from skcomplex.systems import create_simulation

predation = create_simulation('lotka_volterra').prepare().run(until=20.0).trajectory_
signatures = SignatureAnalysis().prepare(predation).run().signatures_
[(item.kind, item.title, item.xscale, list(item.values)) for item in signatures]
```

The signature of each kind of data is this:

- series: the power spectrum and the autocorrelation time, or the return map of a map,
- jump paths: the waiting times and the dispersion index,
- fields: the structure factor of the last frame, with its correlation length or peak wavelength,
- moving agents: the polar and milling orders, or else their nearest-neighbour distances,
- networks: the degree distribution.

#### Complexity and information

Measure how regular, complex, or informative a series is, and how information flows between series.

```python
import math

import numpy as np

from skcomplex.core.analysis import ComplexityAnalysis, EntropyAnalysis, InformationAnalysis

rng = np.random.default_rng(0)
renyi = EntropyAnalysis('renyi', order=2, probabilities=True).prepare(np.full(8, 1 / 8)).run()
renyi.entropy_.round(3).tolist()
sample = EntropyAnalysis('sample', dimension=2).prepare(rng.standard_normal(2000)).run()
sample.entropy_.round(1).tolist()
plane = ComplexityAnalysis(dimension=4).prepare(rng.standard_normal(20_000)).run()
plane.entropy_.round(2).tolist(), plane.complexity_.round(2).tolist()
x = rng.standard_normal(5000)
y = np.concatenate([[0.0], 0.8 * x[:-1]]) + 0.6 * rng.standard_normal(5000)
transfer = InformationAnalysis('transfer', estimator='knn', base=math.e).prepare(np.column_stack([x, y])).run()
transfer.information_[0, 1] > transfer.information_[1, 0]
```

`EntropyAnalysis` measures every entropy by its `kind`:

- `'renyi'` and `'tsallis'` generalize the Shannon entropy of a distribution.
- `'sample'`, `'approximate'`, `'multiscale'`, and `'spectral'` come from `antropy`, and measure the regularity of a
  series.
- `'permutation'` reads ordinal patterns of series with a delay, or of two-dimensional fields in a window, in its
  Shannon, Rényi, or Tsallis `measure`. Discrete states such as spins have many equal values, which `ties='random'`
  ranks at random.
- `'ensemble'` follows the entropy of an ensemble and its divergence from a `reference` law frame by frame, as in the
  H-theorem.

`ComplexityAnalysis` places data on Rosso's complexity-entropy plane, between its bounding curves, with `ordpy`.

The kind `'active-storage'` of `InformationAnalysis` measures the information that the past of a series holds about its
next value. The kind `'transfer'` estimates the flow from every series to every other, by plug-in counts of symbols or
by the nearest-neighbour estimator of `ennemi` on continuous values. Above, `y` follows `x` with a lag, so more
information flows from `x` to `y` than back.

#### Higher-order information and emergence

Tell whether many variables share information redundantly or synergistically, and whether a macro feature has a life of
its own. The measures of many variables take one row per sample and one column per variable.

```python continuation
from skcomplex.core.analysis import PartialInformationAnalysis, SurrogateTestAnalysis

bits = rng.integers(0, 2, (5000, 2))
xor = np.column_stack([bits, bits[:, 0] ^ bits[:, 1]])
round(InformationAnalysis('o-information').prepare(xor).run().information_, 2)
gate = np.column_stack([bits, bits[:, 0] & bits[:, 1]])
parts = PartialInformationAnalysis().prepare(gate).run()
round(parts.redundancy_, 2), round(parts.synergy_, 2)
measure = InformationAnalysis('o-information')
test = SurrogateTestAnalysis(measure, 'information_', n=19, alternative='less', random_state=0)
test.prepare(xor).run().p_value_
```

Among the kinds of `InformationAnalysis`:

- `'total-correlation'` counts the information shared at any order.
- `'dual-total-correlation'` counts the information shared by two or more variables.
- `'o-information'`, their difference, is positive when the variables are redundant and negative when they are
  synergistic, as for the XOR above.
- `'o-information-gradients'` gives the share of every variable.

They take the `'plugin'` estimator of `dit` for symbols, `'gaussian'` for the closed form of a fitted Gaussian, or
`'knn'` for chains of `ennemi` estimates.

- `PartialInformationAnalysis` splits what two sources of symbols tell about a target into redundant, unique, and
  synergistic parts, with a redundancy measure of `dit`.
- `LocalInformationAnalysis` returns the entropy, the mutual information, or the transfer entropy at every sample. Their
  means equal the global measures.
- `SurrogateTestAnalysis` compares the attribute of any of these analyses with its values on copies of the data whose
  variables are shuffled independently.

```python continuation
from skcomplex.core.analysis import EmergenceAnalysis, EpsilonMachineAnalysis

parity = rng.integers(0, 2, (5000, 4))
parity[:, -1] = (parity[:, :-1].sum(axis=1) + np.arange(5000) // 100) % 2
criteria = EmergenceAnalysis('criteria', partition=lambda micro: micro.sum(axis=1) % 2).prepare(parity).run()
criteria.psi_ > 0.5
closure = EmergenceAnalysis('closure', partition=[0, 0, 1, 1]).prepare(rng.integers(0, 4, 5000)).run()
round(closure.gap_, 2)
golden = [1]
for _ in range(20_000):
    golden.append(1 if golden[-1] == 0 else int(rng.random() < 0.5))
machine = EpsilonMachineAnalysis(max_length=4).prepare(golden).run()
len(machine.histories_), round(machine.statistical_complexity_, 2)
```

The kind `'criteria'` of `EmergenceAnalysis` computes the criteria of Rosas et al. (2020) for a macro feature of micro
variables, given by `partition`:

- Ψ is positive when the feature predicts its own future beyond what each micro variable predicts.
- Δ is positive when it predicts a micro variable beyond the micro variables.
- Γ is zero for a feature decoupled from every micro variable.

The kind `'closure'` takes a partition of micro states, as a label per state or a function of the micro series. It
compares how well the macro past and the micro past predict the macro future. A gap of zero means that the
coarse-graining is closed.

`EpsilonMachineAnalysis` reconstructs the causal states of a discrete series by causal-state splitting, with their
transitions, statistical complexity, and entropy rate. The golden mean process above never repeats a 0.

#### Invariants of dynamics

Reconstruct the attractor of a series and measure its dimension, its chaos, and its recurrences.

```python
import numpy as np

from skcomplex.core.analysis import (
    EmbeddingAnalysis,
    ErgodicityAnalysis,
    LyapunovAnalysis,
    RecurrenceAnalysis,
    ScalingAnalysis,
)

points = np.random.default_rng(1).random((3000, 2))
round(ScalingAnalysis('correlation-dimension').prepare(points).run().exponent_, 1)
round(LyapunovAnalysis().prepare([0.906, 0.0, -14.572]).run().kaplan_yorke_dimension_, 2)
chain = ErgodicityAnalysis('ulam', edges=np.linspace(0, 1, 21), samples=200)
chain.prepare(lambda x: 4 * x * (1 - x)).run().density_.round(1)[:3].tolist()
sine = EmbeddingAnalysis(delay=15, dimension=2).prepare(np.sin(np.arange(1000) * 0.1)).run().points_
round(RecurrenceAnalysis().prepare(sine).run().determinism_, 2)
```

- `EmbeddingAnalysis` builds delay coordinates of a series. It puts the delay at the first minimum of the auto mutual
  information, and the dimension where the false nearest neighbours fall below a threshold.
- `ScalingAnalysis('correlation-dimension')` counts close pairs of any point set with a k-d tree, so it serves embedded
  series and agent positions alike.
- The Kaplan-Yorke dimension and the Kolmogorov-Sinai entropy of `LyapunovAnalysis` follow from a Lyapunov spectrum,
  such as the Lorenz spectrum above. `LyapunovAnalysis('largest')` estimates the largest exponent from one series, with
  Rosenstein's method of `nolds`.
- `ErgodicityAnalysis` of kind `'invariant-density'` or `'ulam'` estimates the invariant density of a map.
- `RecurrenceAnalysis` gives the recurrence matrix, rate, determinism, laminarity, line lengths, trapping time, and line
  entropy of embedded series.

#### Surrogates fitted to data

Learn a system from data, then run it, analyse it, or forecast with it like a system you wrote by hand.

```python
import numpy as np

from skcomplex.core.analysis import ModeDecompositionAnalysis, ReservoirAnalysis, SparseIdentificationAnalysis
from skcomplex.core.engine.compile import PassCompiler

times = np.arange(0.0, 10.0, 0.01)
spiral = np.exp(-0.1 * times)[:, None] * np.column_stack([np.cos(times), np.sin(times)])
decomposition = ModeDecompositionAnalysis(['x', 'y'], dt=0.01).prepare(spiral).run()
report = PassCompiler().prepare(decomposition.system_).report_
np.sort(decomposition.eigenvalues_.imag).round(3).tolist(), report.ok
sparse = SparseIdentificationAnalysis(['x', 'y'], degree=1, threshold=0.05, horizon=50, dt=0.01)
sparse = sparse.prepare(spiral).run()
sparse.features_, sparse.coefficients_.round(2).tolist(), sparse.forecast_.shape
reservoir = ReservoirAnalysis(['x', 'y'], horizon=50).prepare(spiral).run()
type(reservoir.system_.time).__name__, reservoir.forecast_.shape
```

`ModeDecompositionAnalysis`, `SparseIdentificationAnalysis`, and `ReservoirAnalysis` prepare on one or more
trajectories or arrays sampled at a uniform step `dt`. Their run sets these attributes:

- `system_`, a `System` of one surrogate element, which a compiler checks and `Simulation` runs like a hand-written
  system,
- `training_error_` and `validation_error_`, the relative errors of the fit on the training samples and on the end of
  every series held out by `validation`,
- `forecast_`, the variables over `horizon` samples after the last one.

The three differ in what they fit:

- `ModeDecompositionAnalysis` fits dynamic mode decomposition with PyDMD. It sets the continuous eigenvalues
  `ln(mu) / dt` in `eigenvalues_`, the modes, their amplitudes, and a surrogate under continuous time with the matrix
  rate law `dg/dt = L g`. A scikit-learn transformer passed as `dictionary`, such as `PolynomialFeatures(2)`, lifts the
  variables to observables first, which is extended DMD. With `inputs`, the names of the variables that are control
  inputs, it fits dynamic mode decomposition with control, `g(t + dt) = A g(t) + B u(t)`, with the `DMDc` of PyDMD. It
  then sets `operator_` and `input_matrix_`, and its surrogate takes every input as a bounded adaptive parameter, which
  a `schedule` of `Simulation` drives. The system `koopman_model_predictive_control` uses these matrices to predict.
- `SparseIdentificationAnalysis` fits sparse identification of nonlinear dynamics with PySINDy. It sets the monomials of
  the variables in `features_`, their coefficients in `coefficients_`, and a surrogate under continuous time with one
  symbolic rate law per variable.
- `ReservoirAnalysis` fits a next-generation reservoir or an echo state network from ReservoirPy, with a ridge readout
  of the increments over one sample. It works on standardized variables unless `standardize` is false. Its surrogate
  runs under discrete time, one step per sample, holds the memory of the reservoir as states, and starts at the last
  sample. So a run of `system_` is the autonomous forecast.

#### Stochastic processes

Measure passage times, Markov chains, fluctuations, irreversibility, and rare events.

```python
import numpy as np

from skcomplex.core.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis, MarkovAnalysis

ring = 2 * np.roll(np.eye(5), 1, axis=1) + np.roll(np.eye(5), -1, axis=1)
stationary = MarkovAnalysis('stationary', matrix='rate').prepare(ring).run().stationary_
production = IrreversibilityAnalysis().prepare(ring).run().entropy_production_
stationary.round(2).tolist(), round(production, 4)
gap = MarkovAnalysis('spectral-gap', matrix='transition').prepare([[0.9, 0.1], [0.2, 0.8]]).run()
round(gap.spectral_gap_, 2)
tilted = LargeDeviationAnalysis(weights=[[1.0]], tilts=np.linspace(-3, 3, 601), values=3.0)
round(float(tilted.prepare([[2.0]]).run().rate_function_[0]), 3)
```

- `PassageAnalysis` measures first passages, absorption, and returns of paths, and the committor and mean first-passage
  time of a transition matrix, by its `kind`. `EventAnalysis('laminar-durations')` measures laminar phases.
- `MarkovAnalysis` estimates transition matrices from paths, and reads transition and rate matrices with `matrix`. It
  gives their stationary laws, spectral gaps, and relaxation times, and Markov state models and implied timescales with
  `pydtmc`.
- `DistributionAnalysis('dispersion-index')`, `ScalingAnalysis('variance-mean')`, and `EventAnalysis('counts')`
  describe fluctuations. `KramersMoyalAnalysis` estimates the drift and diffusion of a series with kernels over its
  increments.
- `IrreversibilityAnalysis` measures entropy production from rates, jump paths, Langevin paths, or transition counts.
  It tests detailed balance and the fluctuation theorem. It checks the thermodynamic uncertainty relation: the squared
  relative fluctuation of a current times the entropy produced is at least 2. It also gives the response to a field and
  the fluctuation-dissipation ratio. The ring above turns one way twice as often as the other, so it produces entropy.
- `LargeDeviationAnalysis` gives the scaled cumulant generating function in three ways: from a tilted generator, from
  blocks of one run, or by cloning a population of trajectories of a system through the steppers of the simulation. The
  last serves systems whose tilted generators are too large to write down. It also gives the Legendre transform at
  `values`, rate functions from histograms at several sizes, and reweighted samples.
- `CorrelationAnalysis('two-time')` and `CollapseAnalysis` measure aging.
- `FitAnalysis` with the laws `'extreme-value'`, `'stable'`, and the named laws such as `'exponential'`, and
  `DistributionAnalysis('records')`, describe extremes and limit laws.

Cloning advances every clone of a population by an interval, weights it by `exp(s A)` for the increment `A` of a
time-additive observable, and resamples the population by the weights. The observable is the dotted name of a state
whose increment is counted, or a function of the trajectory of each interval. A warning names a population whose
effective size collapses, a sign that the estimate is biased.

```python
from typing import Annotated

import numpy as np

from skcomplex.core.analysis import IrreversibilityAnalysis, LargeDeviationAnalysis
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import JumpLaw
from skcomplex.core.components.system import System
from skcomplex.core.specs import StateSpec


class CounterElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.n = n

    def _get_laws(self, s):
        return [JumpLaw(1.0, {s.n: 1})]


counter = System(elements=[CounterElement('counter')])
cloning = LargeDeviationAnalysis(
    'cloning', observable='counter.n', tilts=0.5, population=200, until=10.0, random_state=0
)
cloning.prepare(counter).run().scgf_.round(2).tolist(), round(float(np.exp(0.5) - 1), 2)
currents = np.random.default_rng(0).normal(10.0, np.sqrt(30.0), 5000)
relation = IrreversibilityAnalysis('uncertainty-relation', entropy=10.0, random_state=0)
relation.prepare(currents).run().holds_
```

The counter jumps at rate 1, a Poisson process, so the cloning estimate matches the exact value `exp(s) - 1`.

#### Emergence and ergodicity

Find the coarse-graining at which a Markov chain is most causally effective, and test whether runs explore the same
states.

```python
import numpy as np

from skcomplex.core.analysis import EmergenceAnalysis

noisy = np.zeros((8, 8))
noisy[:7, :7], noisy[7, 7] = 1 / 7, 1.0
round(EmergenceAnalysis('effective-information').prepare(noisy).run().information_, 3)
found = EmergenceAnalysis('causal').prepare(noisy).run()
found.partition_.tolist(), round(found.macro_information_, 3), round(found.gain_, 3)
found.macro_matrix_.tolist()
```

`EmergenceAnalysis` reads a transition matrix for its kinds `'effective-information'` and `'causal'`.

- The effective information measures how much a transition matrix constrains the next state. It is the mutual
  information, in bits, between a uniform intervention on the current state and the next state. It equals the
  determinism of the matrix minus its degeneracy. Rows with no observed transition are left out, with an
  `UnobservedStatesWarning` that names their states.
- The kind `'causal'` reports the gain in effective information of a partition over the micro states. Without a
  partition it searches for one, greedily by merging the pair of blocks with the largest gain, or exhaustively for at
  most 10 states. The partition is an integer label per state or a function of the array of states. The macro matrix
  sums the transitions within every block, weighting the states of a block uniformly or by the stationary distribution.

In the chain of Hoel, Albantakis, and Tononi above, seven noisy states merge into one deterministic macro state, which
carries more effective information than the micro states.

`ErgodicityAnalysis` of kind `'breaking'`, `'metric'`, or `'distance'` diagnoses whether the runs of an ensemble explore
the same states. The ensemble has one row per run, as `ExperimentResult.ensemble` returns it.

```python
import numpy as np

from skcomplex.core.analysis import ErgodicityAnalysis

rng = np.random.default_rng(0)
mixing = rng.standard_normal((100, 5000))
frozen = rng.choice([-1.0, 1.0], size=(100, 1)) + 0.1 * rng.standard_normal((100, 5000))
results = []
for runs in (mixing, frozen):
    breaking = ErgodicityAnalysis('breaking').prepare(runs).run().ergodicity_breaking_
    metric = ErgodicityAnalysis('metric').prepare(runs).run().exponent_
    distance = ErgodicityAnalysis('distance').prepare(runs).run().ergodicity_distance_
    results.append((round(breaking, 2), round(metric, 1), round(distance, 1)))
results
```

- The ergodicity-breaking parameter is the variance of the time averages across runs over the variance of all values.
  It tends to 0 for ergodic runs, and stays near 1 for runs frozen in separate states.
- The fluctuation metric of Thirumalai and Mountain is the variance of the running time averages. It decays as `1 / t`
  for ergodic runs, and levels off for frozen ones.
- The ergodicity distance is the median total-variation distance between the values of one run over time and the values
  of the whole ensemble.

#### Fields and lattices

Predict patterns before a run, and measure domains, wavelengths, clusters, and transitions of fields after it.

```python
import numpy as np

from skcomplex.core.analysis import ClusterAnalysis, DistributionAnalysis, DomainAnalysis, StabilityAnalysis
from skcomplex.systems import create_system

relation = StabilityAnalysis('dispersion').prepare(create_system('brusselator', shape=(128,), b=1.9)).run()
relation.band_.min().round(2), relation.band_.max().round(2)

stripes = np.cos(2 * np.pi * np.arange(64) / 8)[:, None] * np.ones(64)
DomainAnalysis('wavelength').prepare(stripes).run().wavelength_.tolist()
grid = np.random.default_rng(2).random((128, 128)) < 0.7
ClusterAnalysis('percolation').prepare(grid).run().spanning_
samples = np.random.default_rng(3).standard_normal(10_000)
DistributionAnalysis('moment-ratio').prepare(samples).run().cumulant_.round(1).tolist()
```

- `SpectrumAnalysis('structure-factor')` averages the power of a field radially, on a periodic grid of any dimension.
  `CorrelationAnalysis('length')` fits its Ornstein-Zernike form, or the decay of the correlation function.
- `DomainAnalysis` measures coarsening with the kinds `'size'` and `'growth'`, the wavelength and orientation of
  patterns with `'wavelength'`, and the interfaces between domains with `'interface-density'`.
- `ScalingAnalysis('variance-mean')` measures the number fluctuations of density fields in boxes of growing side.
- `ClusterAnalysis` with the kinds `'sizes'`, `'percolation'`, and `'spanning'` describes clusters of grids and
  components of graphs.
- `DistributionAnalysis('moment-ratio')`, with `CollapseAnalysis` of the kinds `'crossings'` and `'scaling'`, locates
  transitions by finite-size scaling. The variance from `DistributionAnalysis('moments', absolute=True)` of an order
  parameter, scaled by the system size, is the susceptibility. Its peak over a temperature sweep of a spin system
  locates the transition too.

`StabilityAnalysis('dispersion')` predicts patterns from the laws of a field system before running it. It linearises
them around a uniform state, and turns every Laplacian into its eigenvalue on each Fourier mode of the lattice. It
returns the growth rate and frequency of every mode, exact for the discretization of the system. A band of growing
modes is a Turing instability, and a growing mode with a nonzero frequency travels.

#### Agents

Measure how agents align, how they spread in space, and how groups segregate.

```python
import numpy as np

from skcomplex.core.analysis import AlignmentAnalysis, CoarseGrainAnalysis, NeighborAnalysis
from skcomplex.systems import create_simulation

flock = create_simulation('boids').prepare().run(until=5).trajectory_
AlignmentAnalysis('mean-vector').prepare(flock).run().order_.shape
positions = np.random.default_rng(4).random((1000, 2))
ratio = NeighborAnalysis('clark-evans', box=[1.0, 1.0]).prepare(positions).run().ratio_
field = CoarseGrainAnalysis([8, 8], box=[1.0, 1.0]).prepare(positions).run().field_
ratio.round(1).tolist(), float(field.sum())
```

- `AlignmentAnalysis` measures the alignment of every frame of agents, of a field of angles or vectors, or of series of
  phases. It uses the length of the mean vector, the mean rotation, the axial order, or the mean phase. The active
  matter systems name them the polar, milling, and nematic orders, and the synchronization systems the Kuramoto order.
- `DistributionAnalysis` and `FitAnalysis` read the speeds of moving agents. So `FitAnalysis(law='maxwell')` in three
  dimensions and `FitAnalysis(law='rayleigh')` in two compare the speeds of a gas, such as the system `ideal_gas`, with
  the Maxwell-Boltzmann law.
- `NeighborAnalysis` describes point patterns in open or periodic boxes by their nearest-neighbour `'distances'`, their
  `'clark-evans'` ratio, and their `'radial-distribution'`.
- `CoarseGrainAnalysis` turns agents into density or polarization fields for the field analyses.
- `NeighborAnalysis('like-share')` and `DomainAnalysis('interface-density')` measure segregation on grids and networks.
  `DistributionAnalysis('total-variation')` gives the distance between the distributions of two groups across units,
  the dissimilarity index of segregation studies.

#### Criticality

Extract avalanches, fit their laws, and estimate how close a process is to criticality.

```python
import numpy as np

from skcomplex.core.analysis import EventAnalysis

found = EventAnalysis('excursions').prepare([0, 1, 3, 2, 0, 0, 4, 1, 0]).run()
found.sizes_.tolist(), found.durations_.tolist()
rng = np.random.default_rng(5)
activity = np.zeros(5000)
activity[0] = 50
for t in range(1, activity.size):
    activity[t] = rng.poisson(0.8 * activity[t - 1] + 10)
round(EventAnalysis('branching').prepare(activity).run().branching_ratio_, 1)
```

- `EventAnalysis('excursions')` extracts avalanches from activity series or relaxation records, and
  `FitAnalysis(law='power')` fits their size and duration laws.
- `EventAnalysis('branching')` estimates the branching ratio by the multistep regression of `mrestimator`, which stays
  unbiased under subsampling. The process above branches with ratio 0.8.
- The exponents `tau_S` of the sizes and `tau_T` of the durations predict the mean size against the duration,
  `<S>(T) ~ T^((tau_T - 1) / (tau_S - 1))`. This is the crackling-noise relation of avalanches, such as those of the
  system `sandpile`.
- `DistributionAnalysis('moments')` gives the order parameter and its variance over a sweep, and
  `ScalingAnalysis('power')` fits the order-parameter exponent.
- `CorrelationAnalysis('overlap')` compares replicas.

#### Network analysis

Measure the structure of a network, its communities, its motifs, and how it looks at coarser scales.

```python
import networkx as nx

from skcomplex.core.analysis import MotifAnalysis, NetworkAnalysis

club = nx.karate_club_graph()
communities = NetworkAnalysis('communities', random_state=0).prepare(club).run()
communities.modularity_.round(2).tolist(), MotifAnalysis().prepare(club).run().counts_[0]['triangle']
```

`NetworkAnalysis` measures the network of a graph, or of every frame of a trajectory, by its `kind`:

- the `'degrees'`, with their moments and tail exponent,
- the `'assortativity'`, the `'centrality'`, the `'neighbor-degree'`, and the `'summary'`,
- the `'communities'`, by Louvain modularity maximization or label propagation, with their agreement with a reference
  partition.

`MotifAnalysis` counts motifs of three and four nodes, and scores them against degree-preserving edge switches or the
configuration model. The system `network_structure_analysis` records a network with these measures.

`RenormalizationAnalysis` coarse-grains a graph, or the network of the last frame of a trajectory with its node states
as attributes. It sets `graphs_`, `degrees_`, `exponents_`, and `clustering_` per scale.

```python
import networkx as nx

from skcomplex.core.analysis import RenormalizationAnalysis

blocks = nx.disjoint_union(nx.complete_graph(20), nx.complete_graph(20))
blocks.add_edge(0, 20)
scale = RenormalizationAnalysis(steps=1).prepare(blocks).run().graphs_[1]
scale.number_of_nodes(), scale.number_of_edges()
```

- The kind `'laplacian'` joins two nodes where the diffusion operator `exp(-time L)` has an entry at least the smaller
  of their diagonal entries, and every component of these joins is a supernode. It runs at `time`, or at the smallest
  peak of the specific heat that merges nodes. Above, the two linked cliques become two linked supernodes.
- The kind `'geometric'` merges `block` consecutive nodes by their `angle`. A supernode sits at their angles weighted by
  `k^max(1, beta)`, with `k` the degree.

In both kinds, two supernodes link when any of their nodes link.

`ScalingAnalysis('allometric')` fits the exponent of a quantity against system size across runs. It reads a mapping
from size to samples, such as the results of a `GridDesign` sweep, or two columns.

```python
import numpy as np

from skcomplex.core.analysis import ScalingAnalysis

rng = np.random.default_rng(0)
samples = {size: size**1.15 * rng.lognormal(0.0, 0.3, 4) for size in np.geomspace(100, 1e5, 50)}
fit = ScalingAnalysis('allometric').prepare(samples).run()
round(fit.exponent_, 1), bool(fit.interval_[0] < 1.15 < fit.interval_[1])
```

It regresses the logarithms of the pairs and sets `exponent_`, with the bootstrap confidence interval `interval_` of the
level `confidence` over `resamples` resamples.

## Run experiments

Sweep parameters, repeat runs, and collect their results in one table.

### Experiments

Run a system many times, over a design of parameters, under scenarios, and with replicates, and collect the results in
one table. Complex systems are studied through many runs, and `Experiment` organises them.

```python
from skcomplex.core.components.specs import ScenarioSpec
from skcomplex.core.experiments import Experiment, GridDesign
from skcomplex.systems import describe

bounds = {'stochastic': [True], 'gamma': [0.1, 0.2]}
sir = describe('sir').factory
ensemble = Experiment(sir, GridDesign(), bounds, replicates=10, until=60, random_state=0).prepare().run()
round(float(ensemble.result_.summary()[0]['I.n.mean']), 1)

policies = [ScenarioSpec('baseline'), ScenarioSpec('quota', schedule=[(50.0, 'harvest.effort', 0.025)])]
harvesting = describe('harvesting').factory
stocks = Experiment(harvesting, scenarios=policies, until=80, outputs=['land/stock.n']).prepare().run()
[round(float(row['land/stock.n'])) for row in stocks.result_.rows]
```

`Experiment` runs a system, or a factory of systems, over three things:

- a design of factory arguments and dotted paths, such as `gamma` above,
- scenarios declared as `ScenarioSpec`, such as a harvest quota that starts at time 50,
- replicates.

Every design row and replicate gets its own random stream, spawned from `random_state` in a fixed order. So results are
the same serially, on threads, or on processes. Every scenario of a row and replicate shares its stream. Overrides of
parameters and initial states reuse one compiled problem, so sweeps of thousands of runs compile once.

An experiment speaks the two verbs of every object. `prepare` validates its settings and resolves the system and the
runs. `run` executes them, and keeps the results in `result_`.

The parts of an experiment are these:

- **Designs.** They are `GridDesign`, `RandomDesign`, `LatinHypercubeDesign`, `SobolDesign`, `RowsDesign`, and the
  designs of sensitivity analysis `OneAtATimeDesign`, `MorrisDesign`, and `SaltelliDesign`. A design stores its
  settings, such as `SobolDesign(n=64)`, and prepares on the `bounds` of the experiment. The bounds are a pair or a
  SciPy distribution per name for a sampled design, a sequence of values per name for a grid, and nothing for a rows
  design. `prepare` sets the rows of a design in `rows_`.
- **Outputs.** `outputs` keeps states and observables as series, after `warmup`, and functions of a trajectory as
  scalars.
- **Analyses.** `analyses` are unprepared analyses. Every run clones each one, prepares it on its trajectory, runs it,
  and keeps what its `run` sets as columns, such as `entropyanalysis.entropy`.
- **Results.** They give one row per run, `series`, `ensemble` with one row per run on common times and the seeds of the
  runs, `summary` with means, standard deviations, and quantiles across replicates, and `to_frame` for pandas.

#### Calibration and sensitivity

Fit the parameters of a system to observed data, and rank which inputs matter most.

```python
import numpy as np

from skcomplex.core.experiments import CalibrationExperiment, Experiment, OneAtATimeDesign, SensitivityAnalysis
from skcomplex.systems import describe

logistic_growth = describe('logistic_growth').factory
times = np.arange(0.0, 21.0)
observed = 500 / (1 + 49 * np.exp(-0.8 * times))
bounds = {'growth.r': (0.1, 2.0), 'growth.K': (100, 1000)}
fit = CalibrationExperiment(logistic_growth, {'population.n': (times, observed)}, bounds).prepare().run()
round(fit.estimates_['growth.r'], 2), round(fit.r2_, 4)

sweep = Experiment(logistic_growth, OneAtATimeDesign(), bounds, until=20, outputs=['population.n'])
sweep = sweep.prepare().run()
SensitivityAnalysis('oat', 'population.n').prepare(sweep.result_).run().ranking_
```

- `CalibrationExperiment` fits dotted paths to observed series, by least squares or a Gaussian likelihood. It keeps the
  estimates and their covariance, with R², RMSE, MAE, and MAPE. Above, it recovers the growth rate 0.8 of the data.
- `SensitivityAnalysis` prepares on the result of an experiment and ranks its inputs one at a time, by Morris
  screening, or by Sobol indices with SALib.

## Extend the library

Add your own elements, interactions, laws, and other parts without changing the library, and see where everything
lives.

### Extending the general classes

Give a general class of the library the words of your field, without losing its checks and laws. A general class names
a mathematical operation, and a subclass names what it means in a field. The subclass overrides hooks, and the checks
and the laws of its parent keep working.

#### Roles and parameters

Rename a role or a parameter of a general interaction, or fix a parameter to a value.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import MovementInteraction, RateInteraction
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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


inflow = System(elements=[StockElement('tank')], interactions=[InflowInteraction('fill', stock='tank')])
round(float(Simulation(inflow, dt=0.1).prepare().run(until=1).trajectory_.values('tank.level')[-1]), 3)
```

`_get_role` renames a role, here `target` to `stock`, in the symbolic form and the agent form alike. `_get_parameter`
renames a parameter in the same way. An argument that the class does not declare as a parameter is a plain value, so a
subclass fixes a parameter by setting it, as `InflowInteraction` fixes the rate to 1.

```python continuation
class BoldWalkInteraction(MovementInteraction):
    def __init__(self, label, *, target: Annotated[str, RoleSpec()], boldness: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label, target=target)
        self.boldness = boldness

    def _get_parameter(self, name):
        return {'probability': 'boldness'}.get(name, name)


class WalkerElement(StockElement):
    def _get_laws(self, s):
        return []


walkers = GroupElement('walkers', element=WalkerElement, size=2, location=[(2, 2)] * 2)
walk = System(
    elements=[walkers],
    interactions=[BoldWalkInteraction('walk', target='walkers')],
    space=LatticeSpace((5, 5)),
    time=DiscreteTime(),
)
len(Simulation(walk, random_state=0).prepare().run(until=2).trajectory_.agent_locations('walkers'))
```

Here the movement probability of `MovementInteraction` becomes the `boldness` of a walker.

#### Variants

Choose the variant of a general interaction with a named value, a spec, or a function. Each argument takes some of
these:

- The pairing of a transfer, a matching, or a grouping takes a `PartnerSpec` or a function.
- The rule of a choice or an imitation, the groups of a grouping, the selection and the replacement of an evolution,
  the action of a topology, and the mode of a cascade, a coupling, or a map take their named values or a spec of a
  subclass.
- A step, a placement, and a move take their named values or a function, and a pick takes its named values. None of
  them takes a spec, because a function already says what a spec would. A spec given to one of them raises an error
  that lists what the argument takes.

Every other value raises the same error, so the checks of the parent keep working.

#### Partners and settlement

Say whom an agent pairs with in a transfer, and how both partners settle the amount.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import TransferInteraction
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import ContinuousSpace, DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
system = System(elements=[eaters, providers], interactions=[meal], time=DiscreteTime())
trajectory = Simulation(system, random_state=0).prepare().run(until=1).trajectory_
trajectory.agents('eaters', 1).values['level'].tolist()
```

`PartnerSpec(candidates, choice, fallback, capacity, score, clearing)` says whom an agent pairs with:

- The candidates are `'all'`, `'neighbors'`, `'cell'`, `'local'` for both of these, a distance, or the name of a state
  that holds a distance per agent. So it works in continuous space too.
- The choice is `'uniform'`, `'score'`, `'nearest'`, or `'all'`.
- The fallback `'nearest'` takes the nearest partner when no one is within reach.

A `settlement(source, target, amounts, system)` of a transfer receives both partners and the drawn amounts, so both ends
credit one draw.

#### Agent states that hold objects

Give an agent a state that holds any value, such as a tuple of orders, and tell analyses what a state means.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import RuleLaw
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
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
queues = System(elements=[group], time=DiscreteTime())
trajectory = Simulation(queues, random_state=0).prepare().run(until=1).trajectory_
trajectory.agents('queues', 1).values['orders'].tolist()
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

### Package layout

Find where every class lives, so you know what to import and where a part of your own belongs. The package has two
parts:

- `skcomplex.core` is the general library: what a system is, how it runs, and how you analyse and experiment with it.
- `skcomplex.systems` is the catalog of ready-made systems, built from the core.

```python
from skcomplex.core.analysis import SymbolicAnalysis
from skcomplex.core.base import BaseConfig
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import Simulation
from skcomplex.core.experiments import Experiment
from skcomplex.systems import describe

[issubclass(cls, BaseConfig) for cls in (System, Simulation, SymbolicAnalysis, Experiment)]
describe('lotka_volterra').field
```

A system, a simulation, an analysis, and an experiment all derive from one root class, `BaseConfig`, and every
ready-made system belongs to a field and a subfield of the catalog.

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

#### The root modules

Every package of the core shares the root modules:

- `skcomplex.core.base` holds `BaseConfig`, the root of every configurable class. It stores the settings of a class
  under their own names, as scikit-learn does, so `get_params`, `set_params`, and `clone` work on every class. It gives
  every class the verbs `prepare` and `run`.
- `skcomplex.core.errors` holds the errors and warnings, such as `ValidationError`, `RoleError`, `SolverError`, and
  `RewritingError`.
- `skcomplex.core.types` holds the types that every layer reads, such as arrays, functions, keys, and kinds.
- `skcomplex.core.specs` holds the specs, frozen values that declare, record, or hold what the classes read. These
  include the specs that annotate states, parameters, and roles, such as `StateSpec`, `ParameterSpec`, and `RoleSpec`,
  the specs of runs and structures, such as `ObservableSpec` and `WienerProcessSpec`, and the specs of rules, domains,
  and geometries, such as `TableSpec`, `LinearRuleSpec`, `SubshiftSpec`, and `GeometrySpec`.

#### The core

`skcomplex.core.components` defines what a system is, and nothing about how it runs. Its modules are listed in their
import order, so a module imports only the modules above it:

- `types`: the types of the components, such as the targets of laws and the pairings and rules of matches.
- `base`: `BasePart`, the base of every part of a system, built on `BaseConfig`.
- `spaces`: the continuous, lattice, network, mesh, and Cayley spaces, their neighbourhoods, and the kinds of time.
- `laws`: the `BaseLaw` base and every law. `laws.utils` holds what laws are written with besides Sage, such as
  `where`, reductions, draws, delays, and calls.
- `elements` and `interactions`: `Element`, `GroupElement`, `FieldElement`, and the other elements, and the binding
  bases `Interaction`, `NeighborInteraction`, `PairInteraction`, and `HyperedgeInteraction`, with the general
  interactions built on them.
- `system`: `System`.
- `specs`: the specs that name classes of the components, as a compilation reads them, such as `ScenarioSpec`.

#### The engine

`skcomplex.core.engine` runs systems. Its packages are listed in their import order too:

- `types`: the types that only the engine reads.
- `problems`: the compiled problems, their layouts of states and parameters, and their numeric functions.
- `backends`: the NumPy, Numba, and JAX backends, which lower expressions to numeric functions.
- `compile`: `compile_system`, the compilers, and the passes that turn a system into a problem.
- `solvers`: the solver contract, `step_count`, and every solver.
- `runtime`: `Simulation`, clocks, observers, and trajectories.
- `viz`: the visuals and the Rerun observer.
- `translators`: the translators of system files of other tools, such as `SystemDynamicsTranslator`.

#### Analysis and experiments

`skcomplex.core.analysis` holds the analyses of systems, trajectories, and data, with the types that only they read in
`skcomplex.core.analysis.types`. `skcomplex.core.experiments` holds the experiments over designs of parameters,
scenarios, and replicates, the designs, sensitivity analysis, and calibration. Its `types` and `specs` hold the rows of
a design and the runs of an experiment.

#### The systems

`skcomplex.systems` is the catalog. You use it through four functions: `catalog`, `describe`, `create_system`, and
`create_simulation`. Behind them, the systems are organised by field, `biology`, `chemistry`, `computer_science`,
`economics`, `engineering`, `mathematics`, `physics`, and `social`, and by subfield inside each field, such as
`physics.mechanics` or `biology.ecology`.

- A subfield package is self-contained, and imports no other subfield.
- A realization that several subfields of one field share lives in the `common` package of that field, such as
  `physics.common`.
- Every system is written with the same public classes of the core as a custom system, so its source also works as a
  pattern.
