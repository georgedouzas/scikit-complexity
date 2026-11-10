# Troubleshooting

This page lists every error of `skcomplex.core.errors`, with the shortest code that raises it, its message, what it
means, and the fix. Search the page for the name of the error a run raised. Each class is a subclass of
[SkcomplexError](#skcomplexerror), and most checks of a system raise a subclass of
[ValidationError](#validationerror).

`simulate` checks a whole system before it compiles and raises one error that lists every problem it found. Each line
of that message names the path of the part, the problem, a fix, and a short code in brackets, such as `[role]`. The
error takes the class of its problems when they share one, and is a `ValidationError` when they differ.

Each section runs on its own. Most use a population of hares that grows at the rate 0.55 a year, or the hare and
lynx model of the [build tutorial](build_a_custom_system/index.md) with the rates of Carpenter (2018).

## SkcomplexError

Every error of the library subclasses `SkcomplexError`, so one `except` clause catches them all. Here a solver for
maps meets a system of rates.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import IterationSolver
from skcomplex.core.errors import SkcomplexError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    simulate(system, until=20, solver=IterationSolver())
except SkcomplexError as error:
    name, message = type(error).__name__, str(error)
name, message
```

The error is a `TagError`, one of the subclasses below. Catch the root class where a program treats every error of
the library the same way, such as a sweep that logs failed runs and goes on. Catch a subclass where the fix depends on
the error. A few checks keep the errors of Python and scikit-learn, such as the `ValueError` of
`set_params` for an unknown name and the `KeyError` of `get_values` for a missing path.

## NotPreparedError

A part runs only after `prepare` has checked it. A law run on its own before it is prepared raises this error.

```python
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import RateLaw
from skcomplex.core.errors import NotPreparedError

x = var('x')
try:
    RateLaw(x, 0.5 * x).run({'x': 10.0})
except NotPreparedError as error:
    message = str(error)
message, RateLaw(x, 0.5 * x).prepare('growth').run({'x': 10.0}).value_
```

The message reads "RateLaw must be prepared before it runs. Call `prepare(...)` first." The fix calls `prepare` first,
and the rate at `x = 10` is then 5.0. `simulate` prepares every part of a system itself, so a run never meets this
error.

## ValidationError

A value that breaks a rule of its part or of the run raises a `ValidationError`, the parent of most errors below.
Here the initial state names the hares by the label of the build tutorial, while the catalog class calls them `prey`.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    simulate(system, until=20, initial_state={'hares.x': 10.0})
except ValidationError as error:
    message = str(error)
message, simulate(system, until=1, dt=1.0, initial_state={'prey.x': 10.0}).get_values('prey.x').round(1).tolist()
```

The message says that `hares.x` is not a state of the system and suggests `prey.x` and `predators.x`. With `prey.x`
the run starts from 10.0 thousand hares and reaches 15.9 thousand after a year.

## ContextError

Some objects prepare against an owner, such as a solver against the problem it advances. Prepared alone, without that
context, they raise this error.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import ContextError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    RK4Solver().prepare()
except ContextError as error:
    message = str(error)
message, simulate(system, until=2, dt=1.0, solver=RK4Solver()).get_values('prey.x').round(1).tolist()
```

The message reads "A solver prepares on a compiled problem. Pass the problem to `prepare`." The fix passes the solver
to `simulate`, which compiles the problem and prepares the solver on it. The run gives 30.0, 46.4, and 68.5 thousand
hares.

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
suggests an `EquationLaw` for its value or a state of order 1. The fix below states the value with an equation.

```python continuation
from skcomplex.core.components.laws import EquationLaw


class FixedLevelElement(LevelElement):
    def _get_laws(self, namespace):
        return [EquationLaw(namespace.x, 2.0)]


simulate(CompositeSystem(elements=[FixedLevelElement('level')]), until=1, dt=1.0).get_values('level.x').tolist()
```

The level holds 2.0 at both frames.

## ConservationError

A run checks the invariants it is given at every frame, and a quantity that drifts further than its tolerance raises
this error. The Lotka and Volterra model keeps `V = b H - d ln H + a L - r ln L` constant along every orbit, and forward
Euler drifts off it.

```python
import numpy as np

from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import EulerSolver
from skcomplex.core.errors import ConservationError
from skcomplex.core.specs import ObservableSpec
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)


def get_constant(t, values):
    prey, predators = values['prey.x'], values['predators.x']
    return 0.024 * prey - 0.80 * np.log(prey) + 0.028 * predators - 0.55 * np.log(predators)


constant = ObservableSpec('constant', function=get_constant)
try:
    simulate(
        system, until=20, dt=1.0, solver=EulerSolver(dt=0.1), observables=[constant], invariants={'constant': 1e-3}
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

A tensor state declared symmetric stores only its independent entries, so its rate must have the same symmetry. Here
a field of strain on a lattice takes an antisymmetric rate.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import SR, matrix

from skcomplex.core.components.elements import Element, FieldElement
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import TensorSymmetryError
from skcomplex.core.specs import StateSpec


class StrainElement(Element):
    def __init__(self, label, *, S: Annotated[object, StateSpec(tensor='symmetric')] = 0.0):
        super().__init__(label)
        self.S = S

    def _get_laws(self, namespace):
        S = namespace.S
        return [RateLaw(S, matrix(SR, [[0, S[0, 0]], [-S[0, 0], 0]]))]


sheet = FieldElement('sheet', element=StrainElement, space=LatticeSpace((4, 4)))
try:
    simulate(CompositeSystem(elements=[sheet]), until=0.1)
except TensorSymmetryError as error:
    message = str(error)
message.split(' on the tensor state ')[-1]
```

The code is `tensor-symmetry`. The message says that the rate law on the tensor state `sheet.S` breaks its symmetry
`'symmetric'` at the entry `[1, 0]`. The fix writes the rate with the symmetry of the state, here with
`symmetric_part`.

```python continuation
from skcomplex.core.components.laws.utils import symmetric_part


class SymmetricStrainElement(StrainElement):
    def _get_laws(self, namespace):
        S = namespace.S
        return [RateLaw(S, symmetric_part(matrix(SR, [[0, S[0, 0]], [-S[0, 0], 0]])) - S)]


sheet = FieldElement('sheet', element=SymmetricStrainElement, space=LatticeSpace((4, 4)))
simulate(CompositeSystem(elements=[sheet]), until=0.1).get_values('sheet.S').shape
```

The run records 11 frames of 16 sites with 3 independent entries each.

## RewritingError

A rewriting system grows a word at every generation, and `max_size` bounds its length. The algae of Lindenmayer
replace `A` by `AB` and `B` by `A`.

```python
from skcomplex.core.components.elements import WordElement
from skcomplex.core.components.laws import ProductionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RewritingError


class AlgaeElement(WordElement):
    law_classes = (ProductionLaw,)

    def _get_laws(self, namespace):
        return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]


try:
    simulate(CompositeSystem(elements=[AlgaeElement('algae', axiom='A', max_size=4)]), until=5)
except RewritingError as error:
    message = str(error)
longer = CompositeSystem(elements=[AlgaeElement('algae', axiom='A', max_size=16)])
message, simulate(longer, until=5).get_values('algae.size').tolist()
```

The message says that the word would reach 5 symbols at generation 3, above its maximum size 4. With `max_size=16`
the sizes run through the Fibonacci numbers 1, 2, 3, 5, 8, and 13.

## ParameterError

A parameter name that the problem does not hold, or a value outside its domain, raises this error. A schedule names
the parameters of the built parts by their path.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ParameterError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    simulate(system, until=20, schedule={'predation.alpha': lambda t: 0.0})
except ParameterError as error:
    message = str(error)
message
```

The message lists the valid names, `growth.r`, `predation.a`, `predation.b`, and `death.d`. The fix is
`schedule={'predation.a': lambda t: 0.0}`. `get_params` of a system lists its model parameters, and `names` of a
trajectory lists the paths of the built parts.

## LabelError

Every part of a system needs a label of its own, because roles, states, and parameters are named by it. Two elements
called `hares` collide.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import LabelError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
births = RateInteraction('births', target='hares', state='x', rate=lambda s: 0.55 * s.target.x)
twins = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(4.0, StateSpec())})
try:
    simulate(CompositeSystem(elements=[hares, twins], interactions=[births]), until=1)
except LabelError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `label`, and the message asks for a unique label of letters, digits, and underscores. The fix renames the
second element, such as to `lynx`.

## RoleError

A role names an element by its label. A typing error, or an element left out of the system, leaves the role bound to
nothing.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RoleError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
hunting = RateInteraction(
    'hunting', target='hares', state='x', sources=['lynx'], rate=lambda s: -0.028 * s.target.x * s.sources[0].x
)
try:
    simulate(CompositeSystem(elements=[hares], interactions=[hunting]), until=1)
except RoleError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `role`. The message says that hunting binds the role `sources` to `lynx`, which is not an element of the
system. The fix adds the element `lynx`, or binds the role to a label the system holds. A role reaches an element of a
nested system by a path such as `forest/foxes`.

## RoleElementError

A role may be bound to an element that exists but does not fit it, because the element is of the wrong kind or lacks
a state the role reads. Here births change the state `n` of hares that carry `x`.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import RoleElementError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
wrong_state = RateInteraction('births', target='hares', state='n', rate=lambda s: 0.55)
try:
    simulate(CompositeSystem(elements=[hares], interactions=[wrong_state]), until=1)
except RoleElementError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `role-element`. The message says that the role `target` of births reads the state `n`, but the element
`hares` carries `x`. The fix is `state='x'`, or a state `n` on the element. The page
[Check a system](check.md#a-role-that-rejects-the-kind-of-its-element) shows the same error for a role that accepts
groups only.

## TagError

The tags of a part say what it needs, such as continuous or discrete time. Parts that disagree on a tag, or a solver
that does not support the tags of a system, raise this error. Here a rate in continuous time meets a map in discrete
time.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import MapInteraction, RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import TagError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
births = RateInteraction('births', target='hares', state='x', rate=lambda s: 0.55 * s.target.x)
lynx = DeclaredElement('lynx', variables={'x': DeclaredVariableSpec(4.0, StateSpec())})
deaths = MapInteraction('deaths', target='lynx', state='x', function=lambda s: 0.45 * s.target.x)
try:
    simulate(CompositeSystem(elements=[hares, lynx], interactions=[births, deaths]), until=1)
except TagError as error:
    message = str(error)
message
```

The message says that `births` has the time continuous and `deaths` the time discrete. The fix writes both parts in
one time, or splits the system into nested systems that run as partitions, as the tutorial
[Simulate](simulate.md#co-simulation) shows. `get_tags` of each part shows where a tag comes from.

## SpaceError

A part that needs a space, a vector state without a space to give its dimension, or a location outside its space
raises this error. Here a group sits on sites but has neither a size nor a space.

```python
from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import SpaceError

try:
    simulate(CompositeSystem(elements=[GroupElement('herds', element=Element, on='sites')]), until=1)
except SpaceError as error:
    message = str(error)
message, GroupElement('herds', element=Element, on='sites', space=LatticeSpace((5, 5))).prepare().size_
```

The message reads "Group 'herds' lives on sites but has neither a size nor a space." The fix gives the group a space,
here a lattice of five by five sites, or a `size`.

## SpaceTooLargeError

A space that would hold more sites than its limit raises this error before it allocates them. The Cayley graph of the
free group on two generators is infinite.

```python
from sage.all__sagemath_gap import libgap

from skcomplex.core.components.spaces import CayleySpace
from skcomplex.core.errors import SpaceTooLargeError

try:
    CayleySpace(libgap.FreeGroup(['a', 'b'])).prepare()
except SpaceTooLargeError as error:
    message = str(error)
message, CayleySpace(libgap.FreeGroup(['a', 'b']), ball=2).prepare().size_
```

The message reads "The group space has more than 100000 sites. Ask for a smaller ball or a finite quotient, or raise
max_sites." The fix `ball=2` keeps the 17 words of length at most two.

## PhaseSpaceTooLargeError

A phase space has one site per cell of its position lattice and of every class axis, such as a velocity, so its size
is the product of all of them. `limit` guards the memory.

```python
from skcomplex.core.components.spaces import LatticeSpace, PhaseSpace
from skcomplex.core.errors import PhaseSpaceTooLargeError
from skcomplex.core.specs import AxisSpec

try:
    PhaseSpace(LatticeSpace((100, 100)), [AxisSpec('v', 64, -4.0, 4.0)], limit=100_000).prepare()
except PhaseSpaceTooLargeError as error:
    message = str(error)
message, PhaseSpace(LatticeSpace((10, 10)), [AxisSpec('v', 64, -4.0, 4.0)], limit=100_000).prepare().size_
```

The message reads "The phase space would take 640000 sites, above its limit of 100000." The fix uses fewer cells,
here a lattice of ten by ten with 6400 sites, or a higher `limit` when the memory allows.

## UnregisteredLawError

Each law class compiles to a mathematical form, such as an ordinary differential equation for `RateLaw`. A law of a
class the engine has no form for raises this error, even a subclass of a core law. Here a growth law subclasses
`RateLaw`.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import UnregisteredLawError
from skcomplex.core.specs import StateSpec


class GrowthLaw(RateLaw):
    pass


class HaresElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 30.0):
        super().__init__(label)
        self.x = x

    def _get_laws(self, namespace):
        return [GrowthLaw(namespace.x, 0.55 * namespace.x)]


try:
    simulate(CompositeSystem(elements=[HaresElement('hares')]), until=1)
except UnregisteredLawError as error:
    message = str(error)
message.splitlines()[-1]
```

The code is `law-unregistered`. The message says that the law class `GrowthLaw` of `hares` has no form the engine
knows. The fix writes the law with a class the core provides, here `RateLaw(namespace.x, 0.55 * namespace.x)`.

## IncompatibleLawsError

The laws of one system must compile to one mathematical form, and a law that holds what its form cannot run raises
this error. Here a rate draws a random number, which only update and renewal laws do.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.laws.utils import normal
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import IncompatibleLawsError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
noisy = RateInteraction('births', target='hares', state='x', rate=lambda s: normal('kick', 0.55, 0.1) * s.target.x)
try:
    simulate(CompositeSystem(elements=[hares], interactions=[noisy]), until=1)
except IncompatibleLawsError as error:
    message = str(error)
message.splitlines()[-1]
```

The message says that the `RateLaw` of births holds the random draw `normal_kick(0.55, 0.1)`. It suggests noise laws
for Wiener processes, jump laws for random events, or renewal laws in continuous time. A nested system whose own parts
mix law kinds raises the same error with the code `laws-mixed`, and the fix moves each kind into a nested system of
its own.

## BackendError

A backend turns laws into numeric functions, and a law it cannot write raises this error. LAMMPS runs particles with
positions and velocities, so it cannot run two populations.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import LammpsSolver
from skcomplex.core.errors import BackendError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    simulate(system, until=1, dt=1.0, solver=LammpsSolver())
except BackendError as error:
    message = str(error)
message
```

The message lists what the LAMMPS backend cannot run, such as the element `prey`, which declares no vector position
state of order 2. The fix leaves out `solver` and `backend`, so the tags choose a pair that runs the laws, or picks
one from the [solvers page](../../parts/solvers.md) and the [backends page](../../parts/backends.md).

## PipelineError

A run does not raise this error. The compiler of the library lowers a system through passes, each of which reads
some fields of the compilation and writes others. A pass that reads a field no pass writes, two passes that write one
field, or passes that read each other in a cycle raise it while the compiler orders them. It concerns those who add a
pass to the compiler, and the fix adds the pass that writes the field or splits the cycle.

## SolverError

A solver that does not fit its problem, or that fails to advance it, raises this error. A fixed step must be positive.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.engine.solvers import RK4Solver
from skcomplex.core.errors import SolverError
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

system = LotkaVolterraSystem(prey=30.0, predators=4.0, r=0.55, a=0.028, b=0.024, d=0.80)
try:
    simulate(system, until=2, dt=1.0, solver=RK4Solver(dt=-0.1))
except SolverError as error:
    message = str(error)
message, simulate(system, until=2, dt=1.0, solver=RK4Solver(dt=0.1)).get_values('prey.x').round(1).tolist()
```

The message reads "dt = -0.1 must be positive." With a step of 0.1 the run gives 30.0, 46.4, and 68.5 thousand hares.
A solver that fails to advance, such as an implicit solve with no solution near the last values, raises the same
class. A smaller step or a stiff solver fixes that case.

## InfeasibleError

A linear program with no feasible point raises this error when its law asks to raise. Two plants of 100 and 50 units
cannot meet a demand of 300.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import LinearProgramLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import InfeasibleError
from skcomplex.core.specs import LinearObjectiveSpec, ParameterSpec, StateSpec

Output = Annotated[float, StateSpec(algebraic=True)]
Demand = Annotated[float, ParameterSpec()]


class GridElement(Element):
    law_classes = (LinearProgramLaw,)

    def __init__(self, label, *, coal: Output = 0.0, gas: Output = 0.0, demand: Demand = 120.0):
        super().__init__(label)
        self.coal, self.gas, self.demand = coal, gas, demand

    def _get_laws(self, namespace):
        coal, gas = namespace.coal, namespace.gas
        constraints = [coal + gas == namespace.demand, coal <= 100, gas <= 50, coal >= 0, gas >= 0]
        return [LinearProgramLaw([coal, gas], constraints, [LinearObjectiveSpec(20 * coal + 50 * gas, sense='min')])]


try:
    simulate(CompositeSystem(elements=[GridElement('grid', demand=300.0)], time=StaticTime()), until=0.0)
except InfeasibleError as error:
    message = str(error)
message
```

The message says that the program of `grid` has no solution at time 0, names the demand constraint and the variables
`grid.coal` and `grid.gas`, and suggests `infeasible='zero'` or `infeasible='hold'`. The fix is a demand the plants can
meet, or one of those policies on `LinearProgramLaw`.

## SimulationError

A simulation used in a state that does not allow an operation raises this error. A recorder holds no trajectory before
a run sends it frames.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import RecorderObserver, simulate
from skcomplex.core.errors import SimulationError
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

hares = DeclaredElement('hares', variables={'x': DeclaredVariableSpec(30.0, StateSpec())})
births = RateInteraction('births', target='hares', state='x', rate=lambda s: 0.55 * s.target.x)
growth = CompositeSystem(elements=[hares], interactions=[births])
recorder = RecorderObserver()
try:
    recorder.trajectory_
except SimulationError as error:
    message = str(error)
simulate(growth, until=2, dt=1.0, observers=[recorder])
message, recorder.trajectory_.get_values('hares.x').round(1).tolist()
```

The message reads "The recorder has not received a problem. Run a simulation that observes it first." After the run
the recorder holds the frames 30.0, 52.0, and 90.1.

## CapacityError

A group whose members appear and disappear on a network holds a fixed number of rows, its `capacity`. A request to
add a member beyond it raises this error. Here a network of three nodes gains one node at every step.

```python
import networkx as nx

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.spaces import DiscreteTime, NetworkSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import CapacityError
from skcomplex.core.specs import TopologyRequestSpec


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
