# Troubleshooting

This page lists every error of `skcomplex.core.errors`, with the shortest code that raises it, its message, what it
means, and the fix. Search the page for the name of the error a run raised. Each class is a subclass of
[SkcomplexError](#skcomplexerror), and most checks of a system raise a subclass of
[ValidationError](#validationerror).

`prepare` checks a whole system before it compiles and raises one error that lists every problem it found. Each line
of that message names the path of the component, the problem, a fix, and a short code in brackets, such as `[role]`. The
error takes the class of its problems when they share one, and is a `ValidationError` when they differ.

Each section runs on its own. Most use a population of hares that grows at the rate 0.55 a year, or the hare and
lynx model of the [build tutorial](build_a_custom_system/index.md) with the rates of Carpenter (2018).

## SkcomplexError

Every error of the library subclasses `SkcomplexError`, so one `except` clause catches them all. Here a solver for
maps meets a system of rates.

```python
from skcomplex.core.engine.solvers import IterationSolver
from skcomplex.core.errors import SkcomplexError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    system.set_params(solver=IterationSolver()).prepare().simulate(until=20)
except SkcomplexError as error:
    name, message = type(error).__name__, str(error)
name, message
```

The error is a `TagError`, one of the subclasses below. Catch the root class where a program treats every error of
the library the same way, such as a sweep that logs failed runs and goes on. Catch a subclass where the fix depends on
the error. A few checks keep the errors of Python and scikit-learn, such as the `ValueError` of
`set_params` for an unknown name and the `KeyError` of `get_values` for a missing path.

## NotPreparedError

A system simulates only after `prepare` has checked and compiled it. A system that simulates before it is prepared
raises this error.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import NotPreparedError
from skcomplex.core.systems import System

hares = Element('hares', values={'x': 30.0})
births = Interaction(
    'births', roles={'target': 'hares'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), 0.55 * s.target.x)]
)
system = System(elements=[hares], interactions=[births])
try:
    system.simulate(until=2, dt=1.0)
except NotPreparedError as error:
    message = str(error)
message, system.prepare().simulate(until=2, dt=1.0).get_values('hares.x').round(1).tolist()
```

The message reads "The system 'system' must be prepared before it simulates. Call `prepare()` first." The fix calls
`prepare` first, and the hares then grow from 30.0 to 52.0 and 90.1 thousand. A system whose parameters change after
`prepare`, such as through `set_params`, raises the same error until it is prepared again.

## ValidationError

A value that breaks a rule of its component or of the run raises a `ValidationError`, the parent of most errors below.
Here the initial state names the hares by the label of the build tutorial, while the catalog class calls them `prey`.

```python
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    system.prepare().simulate(until=20, initial_state={'hares.x': 10.0})
except ValidationError as error:
    message = str(error)
message, system.prepare().simulate(until=1, dt=1.0, initial_state={'prey.x': 10.0}).get_values('prey.x').round(
    1
).tolist()
```

The message says that `hares.x` is not a state of the system and suggests `prey.x` and `predators.x`. With `prey.x`
the run starts from 10.0 thousand hares and reaches 15.9 thousand after a year.

## ContextError

Some objects prepare against an owner, such as a solver against the problem it advances. Prepared alone, without that
context, they raise this error.

```python
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ContextError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    RK4Solver().prepare()
except ContextError as error:
    message = str(error)
message, system.set_params(solver=RK4Solver()).prepare().simulate(until=2, dt=1.0).get_values('prey.x').round(
    1
).tolist()
```

The message reads "A solver prepares on a compiled problem. Pass the problem to `prepare`." The fix gives the solver
to the system as its `solver`, and `prepare` of the system compiles the problem and prepares the solver on it. The run
gives 30.0, 46.4, and 68.5 thousand hares.

## AlgebraicStateError

An algebraic state has no derivative of its own. The equations of its system fix its value at every time, so a rate
law on it has no meaning.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import AlgebraicStateError
from skcomplex.core.specs import StateSpec


class LevelElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec(algebraic=True)] = 0.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, namespace):
        return [RateLaw(namespace.x, 1.0)]


try:
    simulate(CompositeSystem(elements=[LevelElement('level')]), until=1)
except AlgebraicStateError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `algebraic-law`. The message says that the `RateLaw` of `level` targets the algebraic state `level.x`, and
suggests an `ConstraintLaw` for its value or a state of order 1. The fix below states the value with an equation.

```python continuation
from skcomplex.core.components.laws import ConstraintLaw


class FixedLevelElement(LevelElement):
    def _get_laws(self, namespace):
        return [ConstraintLaw(namespace.x, 2.0, target=namespace.x)]


simulate(CompositeSystem(elements=[FixedLevelElement('level')]), until=1, dt=1.0).get_values('level.x').tolist()
```

The level holds 2.0.

## ConservationError

A run checks the invariants it is given at every frame, and a quantity that drifts further than its tolerance raises
this error. The Lotka and Volterra model keeps `V = b H - d ln H + a L - r ln L` constant along every orbit, and forward
Euler drifts off it.

```python
import numpy as np

from skcomplex.core.engine.solvers import EulerSolver
from skcomplex.core.errors import ConservationError
from skcomplex.core.components import Observable
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)


def get_constant(t, values):
    prey, predators = values['prey.x'], values['predators.x']
    return 0.024 * prey - 0.80 * np.log(prey) + 0.028 * predators - 0.55 * np.log(predators)


constant = Observable('constant', function=get_constant)
try:
    system.set_params(solver=EulerSolver(dt=0.1)).prepare().simulate(
        until=20, dt=1.0, observables=[constant], invariants={'constant': 1e-3}
    )
except ConservationError as error:
    message = str(error)
message
```

The message says that the invariant changed from -2.65142 to -2.64349 at frame 1, beyond its tolerance 0.001. The
fix is a smaller step, a solver of higher order such as `RK4Solver`, or the default solver, which keeps the constant.

## TensorShapeError

The operations on tensors read matrices and vectors of expressions. A value of another shape raises this error.

```python
from sage.all__sagemath_symbolics import SR, identity_matrix, var

from skcomplex.core.components.laws.utils import tensor_divergence
from skcomplex.core.errors import TensorShapeError

p = var('p')
try:
    tensor_divergence(p)
except TensorShapeError as error:
    message = str(error)
message, tensor_divergence(p * identity_matrix(SR, 2))
```

The message reads "tensor_divergence needs a square matrix, but got p, which has another shape." The fix passes the
pressure as the isotropic tensor `p I`, whose divergence is `(partial(p, 0), partial(p, 1))`, the gradient of `p`.

## TensorSymmetryError

A value that holds a symmetric tensor, `Annotated[object, Tensor('symmetric')]`, stores only its independent entries,
so its rate must have the same symmetry. Here a field of strain on a lattice takes an antisymmetric rate.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import SR, matrix

from skcomplex.core.components.elements import Element, Tensor
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.errors import TensorSymmetryError
from skcomplex.core.systems import System


class Strain(Element):
    def __init__(self, label, *, S: Annotated[object, Tensor('symmetric')] = 0.0, space=None):
        super().__init__(label, space=space)
        self.S = S


def twist(s):
    S = s.target.S
    return [RateLaw(S, matrix(SR, [[0, S[0, 0]], [-S[0, 0], 0]]))]


sheet = Strain('sheet', space=LatticeSpace((4, 4)))
try:
    System(elements=[sheet], interactions=[Interaction('twist', roles={'target': 'sheet'}, laws=twist)]).prepare()
except TensorSymmetryError as error:
    message = str(error)
message.split(' on the tensor state ')[-1]
```

The code is `tensor-symmetry`. The message says that the rate law on the tensor state `sheet.S` breaks its symmetry
`'symmetric'` at the entry `[1, 0]`. The fix writes the rate with the symmetry of the state, here with
`symmetric_part`.

```python continuation
from skcomplex.core.components.laws.utils import symmetric_part


def relax(s):
    S = s.target.S
    return [RateLaw(S, symmetric_part(matrix(SR, [[0, S[0, 0]], [-S[0, 0], 0]])) - S)]


system = System(elements=[sheet], interactions=[Interaction('relax', roles={'target': 'sheet'}, laws=relax)])
system.prepare().simulate(until=0.1).get_values('sheet.S').shape
```

The run records 11 frames of 16 sites with 3 independent entries each.

## RewritingError

A rewriting system grows a word at every generation, and `max_size` bounds its length. The algae of Lindenmayer
replace `A` by `AB` and `B` by `A`.

```python
from skcomplex.core.errors import RewritingError
from skcomplex.core.systems import System
from skcomplex.systems.general.rewriting import ProductionLaw, RewritingInteraction, WordElement

growth = RewritingInteraction('growth', target='algae', laws=[ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')])
try:
    System(elements=[WordElement('algae', axiom='A', max_size=4)], interactions=[growth]).prepare().simulate(until=5)
except RewritingError as error:
    message = str(error)
longer = System(elements=[WordElement('algae', axiom='A', max_size=16)], interactions=[growth]).prepare()
message, longer.simulate(until=5).get_values('algae.size').tolist()
```

The message says that the word would reach 5 symbols at generation 3, above its maximum size 4. With `max_size=16`
the sizes run through the Fibonacci numbers 1, 2, 3, 5, 8, and 13.

## ParameterError

A parameter name that the problem does not hold, or a value outside its domain, raises this error. A schedule names
the parameters of the built components by their path.

```python
from skcomplex.core.errors import ParameterError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    system.prepare().simulate(until=20, schedule={'predation.alpha': lambda t: 0.0})
except ParameterError as error:
    message = str(error)
message
```

The message lists the valid names, `growth.r`, `predation.a`, `predation.b`, and `death.d`. The fix is
`schedule={'predation.a': lambda t: 0.0}`. `get_params` of a system lists its model parameters, and `names` of a
trajectory lists the paths of the built components.

## ParameterShapeError

An array argument of an interaction is a parameter when it is a vector, a square matrix of 2 or 3 rows, or an array
with one axis per role whose group has as many members, such as the coupling matrix of a group of oscillators. An
array that no group shapes, such as a matrix of four rows read by the laws of one element, raises this error.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import total
from skcomplex.core.errors import ParameterShapeError
from skcomplex.core.types import Role
from skcomplex.core.systems import System


class Coupling(Interaction):
    def __init__(self, label, *, target: Role = 'nodes', a=1.0):
        super().__init__(label)
        self.target = target
        self.a = a

    def _prepare(self, context):
        super()._prepare(context)
        x = context.target.x
        self.laws_ = [RateLaw(x, total(x, weights=context.a) - x)]


coupling = Coupling('coupling', a=np.ones((4, 4)) / 4)
try:
    System(elements=[Element('nodes', values={'x': 1.0})], interactions=[coupling]).prepare()
except ParameterShapeError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `parameter-shape`. The message names `coupling.a` and its shape `(4, 4)`. The fix gives the element one
member per row and column, so the group of four nodes shapes the matrix.

```python continuation
nodes = Element('nodes', values={'x': [1.0, 2.0, 3.0, 4.0]}, size=4)
System(elements=[nodes], interactions=[coupling]).prepare().form_
```

The system prepares as ordinary differential equations.

## LabelError

Every component of a system needs a label of its own, because roles, states, and parameters are named by it. Two
elements called `hares` collide.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import LabelError
from skcomplex.core.systems import System

hares = Element('hares', values={'x': 30.0})
twins = Element('hares', values={'x': 4.0})
births = Interaction(
    'births', roles={'target': 'hares'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), 0.55 * s.target.x)]
)
try:
    System(elements=[hares, twins], interactions=[births]).prepare()
except LabelError as error:
    message = str(error)
message
```

The message reads "Label 'hares' is used by more than one child of the root system." The fix renames the second
element, such as to `lynx`, with a label of letters, digits, and underscores.

## RoleError

A role names an element by its label. A typing error, or an element left out of the system, leaves the role bound to
nothing.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import RoleError
from skcomplex.core.systems import System


def hunt(s):
    return [RateLaw(s.target.x.diff(s.t), -0.028 * s.target.x * s.source.x)]


hares = Element('hares', values={'x': 30.0})
hunting = Interaction('hunting', roles={'target': 'hares', 'source': 'lynx'}, laws=hunt)
try:
    System(elements=[hares], interactions=[hunting]).prepare()
except RoleError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `role`. The message says that hunting binds the role `source` to `lynx`, which is not an element of the
system. The fix adds the element `lynx`, or binds the role to a label the system holds. A role reaches an element of a
nested system by a path such as `forest/foxes`.

## RoleElementError

A role may be bound to an element that exists but does not fit it, because the element is of the wrong kind or lacks
a value the laws read. Here births change the value `n` of hares that carry `x`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import RoleElementError
from skcomplex.core.systems import System

hares = Element('hares', values={'x': 30.0})
wrong_state = Interaction('births', roles={'target': 'hares'}, laws=lambda s: [RateLaw(s.target.n.diff(s.t), 0.55)])
try:
    System(elements=[hares], interactions=[wrong_state]).prepare()
except RoleElementError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `role-element`. The message says that the role `target` of births reads the state `n`, but the element
bound to it carries `x`. The fix reads `s.target.x`, or gives the element a value `n`. The page
[Check a system](check.md#a-role-that-rejects-the-kind-of-its-element) shows the same error for a role that accepts
groups only.

## TagError

The tags of a system say what it is, such as deterministic or stochastic, and what its time is. An experiment or a
solver that does not fit the tags of a system raises this error. Here an ensemble of runs, which compares runs that
draw different random numbers, meets a deterministic system.

```python
from skcomplex.core.errors import TagError
from skcomplex.core.experiments import ensemble
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    ensemble(system, replicates=2, until=1)
except TagError as error:
    message = str(error)
message
```

The message says that `ensemble` needs the randomness stochastic, but `LotkaVolterraSystem` has the randomness
deterministic. Its runs repeat one another, so the fix runs it once with `simulate`, or adds a component that draws,
such as a noise law. `get_tags` of a system shows its tags. Laws whose kinds imply different times, such as a rate and
an update, raise an [IncompatibleLawsError](#incompatiblelawserror) instead.

## SpaceError

A space that cannot hold what it is asked to, a component that needs another kind of space, or a location outside its
space raises this error. Here agents start in a named region of a map, and the map has three dimensions where regions
need two.

```python
from shapely.geometry import box

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.errors import SpaceError
from skcomplex.core.systems import System


def age(agent, system):
    agent.age += 1


def run(extent):
    space = ContinuousSpace(extent=extent, regions={'park': box(0, 0, 2, 2)})
    agents = Element('agents', values={'region': 'park', 'age': 0}, size=10, space=space)
    aging = Interaction('aging', roles={'target': 'agents'}, laws=lambda s: [UpdateLaw(s.target, age)])
    return System(elements=[agents], interactions=[aging]).prepare().simulate(until=1, random_state=0)


try:
    run(extent=[10.0, 10.0, 10.0])
except SpaceError as error:
    message = str(error)
message, int(run(extent=[10.0, 10.0]).get_values('agents.count')[-1])
```

The message reads "The layers park hold regions of a map, and the space has 3 dimensions. Give the space 2
dimensions." The fix gives the space an extent of two dimensions, and the 10 agents then start in the park.

## SpaceTooLargeError

A space that would hold more sites than its limit raises this error before it allocates them. The Cayley graph of a
group in the automata of `skcomplex.systems.computer_science.automata` raises it past 100000 sites, such as the ball of
radius eleven of the free group in `FreeGroupAutomatonSystem(radius=11)`, and a phase space raises its subclass
`PhaseSpaceTooLargeError`. Catching this error catches both. Here a phase space of fifty by fifty positions and 32
speeds is too large for its limit.

```python
from skcomplex.core.components.spaces import AxisSpec, LatticeSpace, PhaseSpace
from skcomplex.core.errors import SpaceTooLargeError

speeds = [AxisSpec('v', 32, -4.0, 4.0)]
try:
    PhaseSpace(LatticeSpace((50, 50)), speeds, limit=10_000).prepare()
except SpaceTooLargeError as error:
    message = str(error)
message, PhaseSpace(LatticeSpace((10, 10)), speeds, limit=10_000).prepare().size_
```

The message reads "The phase space would take 80000 sites, above its limit of 10000." The fix uses fewer cells, here a
lattice of ten by ten with 3200 sites. For the ball of a group, the fix is a smaller radius or a finite quotient.

## PhaseSpaceTooLargeError

A phase space has one site per cell of its position lattice and of every class axis, such as a velocity, so its size
is the product of all of them. `limit` guards the memory.

```python
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.errors import PhaseSpaceTooLargeError
from skcomplex.core.components.spaces import AxisSpec

try:
    PhaseSpace(LatticeSpace((100, 100)), [AxisSpec('v', 64, -4.0, 4.0)], limit=100_000).prepare()
except PhaseSpaceTooLargeError as error:
    message = str(error)
message, PhaseSpace(LatticeSpace((10, 10)), [AxisSpec('v', 64, -4.0, 4.0)], limit=100_000).prepare().size_
```

The message reads "The phase space would take 640000 sites, above its limit of 100000." The fix uses fewer cells,
here a lattice of ten by ten with 6400 sites, or a higher `limit` when the memory allows.

## UnregisteredLawError

Each kind of law compiles to a mathematical form, such as an ordinary differential equation for `RateLaw`. A law of a
class the engine has no form for raises this error, even a subclass of a core law. Here a growth law subclasses
`RateLaw`.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import UnregisteredLawError
from skcomplex.core.systems import System


class GrowthLaw(RateLaw):
    pass


hares = Element('hares', values={'x': 30.0})
births = Interaction(
    'births', roles={'target': 'hares'}, laws=lambda s: [GrowthLaw(s.target.x.diff(s.t), 0.55 * s.target.x)]
)
try:
    System(elements=[hares], interactions=[births]).prepare()
except UnregisteredLawError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `law-unregistered`. The message says that the law class `GrowthLaw` has no form the engine knows. The fix
writes the law with a kind the core provides, here `RateLaw(s.target.x.diff(s.t), 0.55 * s.target.x)`.

## IncompatibleLawsError

The laws of one system must compile to one mathematical form, and a law that holds what its form cannot run raises
this error. Here a rate draws a random number, which only update and renewal laws do.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import normal
from skcomplex.core.errors import IncompatibleLawsError
from skcomplex.core.systems import System


def grow(s):
    return [RateLaw(s.target.x.diff(s.t), normal('kick', 0.55, 0.1) * s.target.x)]


hares = Element('hares', values={'x': 30.0})
try:
    System(elements=[hares], interactions=[Interaction('births', roles={'target': 'hares'}, laws=grow)]).prepare()
except IncompatibleLawsError as error:
    message = str(error)
message.splitlines()[-1]
```

The message says that the `RateLaw` of births holds the random draw `normal_kick(0.55, 0.1)`. It suggests noise laws for
Wiener processes, jump laws for random events, or renewal laws in continuous time. A system whose own laws mix kinds
that imply different times, such as a rate and an update, raises the same error with the code `laws-mixed`, and the
fix moves each kind into a nested system of its own, as the tutorial [Simulate](simulate.md#co-simulation) shows.

## BackendError

A backend turns laws into numeric functions, and a law it cannot write raises this error. LAMMPS runs particles with
positions and velocities, so it cannot run two populations.

```python
from skcomplex.core.engine.solvers import LammpsSolver
from skcomplex.core.errors import BackendError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    system.set_params(solver=LammpsSolver()).prepare().simulate(until=1, dt=1.0)
except BackendError as error:
    message = str(error)
message
```

The message lists what the LAMMPS backend cannot run, such as the element `prey`, which declares no vector position
state of order 2. The fix leaves out `solver` and `backend`, so the tags choose a pair that runs the laws, or picks
one from the [solvers page](../../components/solvers.md) and the [backends page](../../components/backends.md).

## PipelineError

A run does not raise this error. The compiler of the library lowers a system through passes, each of which reads
some fields of the compilation and writes others. A pass that reads a field no pass writes, two passes that write one
field, or passes that read each other in a cycle raise it while the compiler orders them. It concerns those who add a
pass to the compiler, and the fix adds the pass that writes the field or splits the cycle.

## SolverError

A solver that does not fit its problem, or that fails to advance it, raises this error. A fixed step must be positive.

```python
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import SolverError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    system.set_params(solver=RK4Solver(dt=-0.1)).prepare().simulate(until=2, dt=1.0)
except SolverError as error:
    message = str(error)
message, system.set_params(solver=RK4Solver(dt=0.1)).prepare().simulate(until=2, dt=1.0).get_values('prey.x').round(
    1
).tolist()
```

The message reads "dt = -0.1 must be positive." With a step of 0.1 the run gives 30.0, 46.4, and 68.5 thousand hares.
A solver that fails to advance, such as an implicit solve with no solution near the last values, raises the same
class. A smaller step or a stiff solver fixes that case.

## InfeasibleError

A linear program with no feasible point raises this error when its law asks to raise. Two plants of 100 and 50 units
cannot meet a demand of 300.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw, ObjectiveLaw
from skcomplex.core.errors import InfeasibleError
from skcomplex.core.systems import System


def dispatch(s):
    coal, gas = s.target.coal, s.target.gas
    return [
        ObjectiveLaw(20 * coal + 50 * gas, [coal, gas]),
        ConstraintLaw(coal + gas == s.target.demand, target=coal),
        ConstraintLaw(coal <= 100, target=coal),
        ConstraintLaw(gas <= 50, target=gas),
        ConstraintLaw(coal >= 0, target=coal),
        ConstraintLaw(gas >= 0, target=gas),
    ]


def run(demand):
    grid = Element('grid', values={'coal': 0.0, 'gas': 0.0, 'demand': demand})
    interaction = Interaction('dispatch', roles={'target': 'grid'}, laws=dispatch)
    return System(elements=[grid], interactions=[interaction]).prepare().simulate(until=0.0)


try:
    run(demand=300.0)
except InfeasibleError as error:
    message = str(error)
message, run(demand=120.0).get_values('grid.gas').tolist()
```

The message says that the program of `grid` has no solution at time 0, names the demand constraint and the variables
`grid.coal` and `grid.gas`, and suggests `infeasible='zero'` or `infeasible='hold'`. The fix is a demand the plants can
meet, or one of those policies on the `ObjectiveLaw`. A demand of 120 is met by 100 units of coal, the cheaper fuel,
and 20 of gas.

## SimulationError

A simulation used in a state that does not allow an operation raises this error. A recorder holds no trajectory before
a run sends it frames.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.engine.runtime import RecorderObserver
from skcomplex.core.errors import SimulationError
from skcomplex.core.systems import System

hares = Element('hares', values={'x': 30.0})
births = Interaction(
    'births', roles={'target': 'hares'}, laws=lambda s: [RateLaw(s.target.x.diff(s.t), 0.55 * s.target.x)]
)
growth = System(elements=[hares], interactions=[births]).prepare()
recorder = RecorderObserver()
try:
    recorder.trajectory_
except SimulationError as error:
    message = str(error)
growth.simulate(until=2, dt=1.0, observers=[recorder])
message, recorder.trajectory_.get_values('hares.x').round(1).tolist()
```

The message reads "The recorder has not received a problem. Run a simulation that observes it first." After the run
the recorder holds the frames 30.0, 52.0, and 90.1.

## CapacityError

A group without a `capacity` grows as a Python list does, so it never raises this error. A group given a `capacity`
holds that many rows, and a request to add a member beyond it raises this error. Here a network of three nodes gains
one node at every step.

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import CapacityError
from skcomplex.core.components.laws import TopologyRequestSpec


class GrowthElement(Element):
    def _get_laws(self, namespace):
        return [UpdateLaw(TopologyRequestSpec('add-node'), 1)]


def grow(capacity):
    nodes = GroupElement('nodes', element=Element, on='sites', variable=True, capacity=capacity)
    links = GroupElement('links', element=Element, on='edges', variable=True, capacity=3 * capacity)
    network = CompositeSystem(
        elements=[nodes, links, GrowthElement('growth')], space=NetworkSpace(nx.complete_graph(3)), time=DiscreteTime()
    )
    return simulate(network, until=3, random_state=0)


try:
    grow(capacity=4)
except CapacityError as error:
    message = str(error)
message, grow(capacity=6).get_network(-1).nodes.size
```

The message says that the group `nodes` holds 4 members, its capacity, and asks for a larger `capacity`. With room for
6 the network ends with 6 nodes after three steps.

## ExperimentError

An experiment that cannot run its tasks raises this error. A factory of systems must return a system, and one that
forgets to return it gives `None`.

```python
from skcomplex.core.errors import ExperimentError
from skcomplex.core.experiments import sweep
from skcomplex.systems.biology.ecology import LotkaVolterraSystem


def build(r):
    LotkaVolterraSystem(prey=30.0, predators=4.0, r=r, a=0.028, b=0.024, d=0.80)


try:
    sweep(build, {'r': [0.45, 0.65]}, until=1)
except ExperimentError as error:
    message = str(error)
message
```

The message reads "The factory of an experiment returns a system, not NoneType." The fix adds `return` before the
system. An experiment also raises this error when its worker processes die, for example for lack of memory, and the
fix there is fewer workers or a smaller design.

## AnalysisError

An analysis that does not apply to its source raises this error. The equations of Lotka and Volterra have no closed
form, so the exact solution fails.

```python
from skcomplex.core.analysis import symbolic
from skcomplex.core.errors import AnalysisError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    symbolic(system, kind='solve')
except AnalysisError as error:
    message = str(error)
message
```

The message says that no closed form was found for `prey.x` and `predators.x`, and suggests simulating the system
instead. `applicable_analyses` lists the analyses that apply to a source, as the tutorial
[Analyse](analyse.md#how-an-analysis-works) shows.

## EnumerationLimitError

An exact enumeration that does not finish within its time limit raises this error. The elementary modes of a reaction
network, its smallest pathways at steady state, grow in number combinatorially. Here every pair of 14 species
exchanges matter in both directions.

```python
import networkx as nx
import numpy as np

from skcomplex.core.analysis import elementary_modes
from skcomplex.core.errors import EnumerationLimitError


def get_exchanges(size):
    arcs = list(nx.complete_graph(size, create_using=nx.DiGraph).edges)
    stoichiometry = np.zeros((size, len(arcs)))
    for column, (tail, head) in enumerate(arcs):
        stoichiometry[tail, column], stoichiometry[head, column] = -1, 1
    return stoichiometry


try:
    elementary_modes(get_exchanges(14), timeout=0.2)
except EnumerationLimitError as error:
    message = str(error)
message, elementary_modes(get_exchanges(3)).modes.shape
```

The message says that the enumeration over 182 variables did not finish within 0.2 s, and suggests closing exchanges
or enumerating a subnetwork. The fix is a smaller network, or a longer `timeout`. Three species give 5 modes over 6
exchanges: the three pairs that trade back and forth and the two cycles through all three.

## MissingExtraError

Some ready-made systems need an optional extra, a group of packages installed apart from the library. A system of
metabolism built without its extra raises this error when it imports the missing package, with a message such as the
one below.

```text
The system flux_balance needs the extra 'metabolism'. Install it with `pip install 'scikit-complexity[metabolism]'`.
```

The message names the system and the extra and gives the command that installs it. The fix runs the command. The page
[Select a ready-made system](select_a_ready_made_system.md#optional-extras) lists the extras.

## RegistryError

A run does not raise this error. The catalog of ready-made systems reads the class variables of every system module
without importing it, and `all_systems` raises this error when a system class is not named with the suffix `System`,
gives its field path as no string, or computes a listed class variable at run time. It concerns those who add a system
to the catalog, and the fix writes those class variables as literals, as the page [Share a system](share.md#the-system-class) shows.
