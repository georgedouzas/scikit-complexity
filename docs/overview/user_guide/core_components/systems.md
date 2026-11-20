# Systems

A system composes elements, interactions, nested systems, and a space into one model. Elements hold the values,
interactions set the laws that change them, and a space places the members of an element. A `System` takes these
components as data, `prepare` compiles them into one problem, and `simulate` runs it, as `fit` and `predict` do for a
scikit-learn estimator. A `BaseSystem` builds its system from the parameters of a model, so users can set, sweep, and
calibrate them. `RunSettings` names the run settings that suit a system class. [How the components fit](index.md)
shows the families together, and the [systems page](../../../components/systems.md) compares the system classes and
holds their fact sheets.

## System

A `System` holds the components of a model as they are given. Here prey grow and predators hunt them on every site of
a lattice of three by three sites, and prey move between neighbouring sites. The prey start unevenly, with more on the
first site and fewer on the third.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.laws.utils import neighbors
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.systems import System

prey = [1.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
habitats = Element('habitats', values={'prey': prey, 'predators': 0.5}, space=LatticeSpace((3, 3)))
births = Interaction(
    'births',
    roles={'target': 'habitats'},
    laws=lambda s: [RateLaw(s.target.prey.diff(s.t), s.target.prey * (1.0 - s.target.predators))],
)
hunting = Interaction(
    'hunting',
    roles={'target': 'habitats'},
    laws=lambda s: [RateLaw(s.target.predators.diff(s.t), s.target.predators * (s.target.prey - 1.0))],
)
dispersal = Interaction(
    'dispersal',
    roles={'target': 'habitats'},
    laws=lambda s: [RateLaw(s.target.prey.diff(s.t), neighbors(s.target.other.prey - s.target.prey))],
)
ecosystem = System('ecosystem', elements=[habitats], interactions=[births, hunting, dispersal]).prepare()
ecosystem.simulate(until=4, dt=2.0).get_values('habitats.prey').mean(axis=1).round(2).tolist()
```

The mean prey over the lattice rises to 1.65 and falls to 0.52, a cycle of Lotka and Volterra. The element on the
lattice holds one member per site, and its value `predators` starts at 0.5 on every site.

The same class holds a savings account. Interest of 5 percent a year grows the balance, and withdrawals of 120 a
year draw it down.

```python continuation
account = Element('account', values={'x': 1000.0})
interest = Interaction(
    'interest',
    roles={'target': 'account'},
    laws=lambda s: [RateLaw(s.target.x.diff(s.t), 0.05 * s.target.x)],
)
withdrawals = Interaction(
    'withdrawals',
    roles={'target': 'account'},
    laws=lambda s: [RateLaw(s.target.x.diff(s.t), -120.0)],
)
savings = System('savings', elements=[account], interactions=[interest, withdrawals])
savings.prepare().simulate(until=10, dt=5.0).get_values('account.x').round(2).tolist()
```

The withdrawals exceed the interest, so the balance falls to 602.36 after five years and to 91.79 after ten.

**How the engine runs it.** `prepare` checks every component, binds the interactions to the elements their roles name,
and compiles the laws of every component into one problem. The kinds of the laws choose the form of the problem and
its default solver, and `prepare` sets the attributes it learns, such as `form_` and `solver_`.

```python continuation
ecosystem.form_, type(ecosystem.solver_).__name__
```

The rate laws of the ecosystem make ordinary differential equations, so `ScipyODESolver` integrates it.

**How it combines.** A system nests other systems through `systems`, and an interaction names an element of a nested
system by its path, such as `savings/account`. Here a family fund grows by 10 percent a year of the balance of the
nested account, a matching gift that the account itself does not pay.

```python continuation
fund = Element('fund', values={'x': 0.0})
deposit = Interaction(
    'deposit',
    roles={'target': 'fund', 'source': 'savings/account'},
    laws=lambda s: [RateLaw(s.target.x.diff(s.t), 0.1 * s.source.x)],
)
family = System('family', elements=[fund], systems=[savings], interactions=[deposit]).prepare()
trajectory = family.simulate(until=2, dt=1.0)
trajectory.get_values('savings/account.x').round(2).tolist(), trajectory.get_values('fund.x').round(2).tolist()
```

The account falls to 928.22 and 852.76 under its own laws, and the fund grows to 96.44 and 185.52. A nested system
may also be a `BaseSystem` subclass, which the system replaces by the system it builds.

**Limits and errors.** Every role must name an element of the system, and `prepare` raises a `RoleError`, a kind of
`ValidationError`, for a path that names none.

```python continuation
from skcomplex.core.errors import ValidationError

typo = Interaction(
    'interest',
    roles={'target': 'acount'},
    laws=lambda s: [RateLaw(s.target.x.diff(s.t), 0.05 * s.target.x)],
)
try:
    System(elements=[account], interactions=[typo]).prepare()
except ValidationError as error:
    message = str(error)
message
```

The message says that the interaction `interest` binds the role `target` to `acount`, which is not an element or
group path of the system. The fix is the label of the element, `account`.

**A field realization.** The catalog writes a system as a `BaseSystem` subclass whose `_prepare` sets the `System` of
its components, as the next section shows.

See its [API entry][skcomplex.core.systems.System].

## BaseSystem

A `BaseSystem` is the base of a system class with model parameters. The subclass takes the parameters as arguments of
its `__init__` and stores each under the same name, as a scikit-learn estimator does, and its `_prepare` sets
`system_`, the `System` of its components. Class variables such as `until` and `dt` are its tuned run settings. Here an
outbreak grows logistically in a town, at the rate `beta` among `population` people.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.systems import BaseSystem, System


class OutbreakSystem(BaseSystem):
    until = 30.0
    dt = 10.0

    def __init__(self, label='outbreak', beta=0.3, population=1000.0):
        super().__init__(label)
        self.beta = beta
        self.population = population

    def _prepare(self, context=None):
        super()._prepare(context)
        infected = Element('infected', values={'x': 10.0})
        growth = lambda s: self.beta * s.target.x * (1 - s.target.x / self.population)
        spread = Interaction(
            'spread',
            roles={'target': 'infected'},
            laws=lambda s: [RateLaw(s.target.x.diff(s.t), growth(s))],
        )
        self.system_ = System(self.label, elements=[infected], interactions=[spread])


outbreak = OutbreakSystem()
outbreak.get_params(), outbreak.prepare().simulate().get_values('infected.x').round(1).tolist()
```

`simulate` takes the end time 30 and the step 10 from the class. The infected number 168.7 after ten days, 803.0
after twenty, and 987.9 after thirty.

In physics a cup of coffee cools toward the temperature of the room by the law of Newton. The same pattern gives it a
cooling constant and a room temperature as parameters, and `set_params` changes one of them.

```python continuation
class CoolingSystem(BaseSystem):
    until = 20.0
    dt = 10.0

    def __init__(self, label='cup', k=0.1, room=20.0):
        super().__init__(label)
        self.k = k
        self.room = room

    def _prepare(self, context=None):
        super()._prepare(context)
        cup = Element('coffee', values={'T': 90.0})
        newton = lambda s: -self.k * (s.target.T - self.room)
        cooling = Interaction(
            'cooling',
            roles={'target': 'coffee'},
            laws=lambda s: [RateLaw(s.target.T.diff(s.t), newton(s))],
        )
        self.system_ = System(self.label, elements=[cup], interactions=[cooling])


cup = CoolingSystem()
before = cup.prepare().simulate().get_values('coffee.T').round(1).tolist()
after = cup.set_params(k=0.05).prepare().simulate().get_values('coffee.T').round(1).tolist()
before, after
```

With `k = 0.1` the coffee falls from 90 to 45.8 and 29.5 degrees, and with `k = 0.05` it cools more slowly, to 62.5 and
45.8 degrees.

**How the engine runs it.** `prepare` calls `_prepare`, which builds the components from the current parameters, and
prepares `system_`. A parameter set after `prepare` asks for `prepare` again before the next run, and `clone` copies
the parameters, as for any scikit-learn estimator.

```python continuation
from sklearn.base import clone

copy = clone(cup)
copy.get_params(), type(copy.prepare().system_).__name__
```

The clone has the cooling constant 0.05 that `set_params` gave the cup, and its components form a `System`.

**How it combines.** A system class nests inside a `System` through `systems`, and its states are named by its path,
such as `outbreak/infected.x`. The parameters are what calibration fits and what experiments sweep.

```python continuation
town = System('town', systems=[OutbreakSystem()]).prepare()
town.simulate(until=10, dt=10.0).get_values('outbreak/infected.x').round(1).tolist()
```

The nested outbreak reaches 168.7 infected after ten days, as it does alone.

**Limits and errors.** A system simulates only once it is prepared, and again after a parameter changes, so a run
raises a `NotPreparedError` that names the parameter.

```python continuation
from skcomplex.core.errors import NotPreparedError

try:
    outbreak.set_params(beta=0.5).simulate()
except NotPreparedError as error:
    message = str(error)
message
```

The message says that the parameter `beta` of the system changed since it was prepared and asks to call `prepare()`
again.

**A field realization.** The subclasses above are field realizations, an outbreak of epidemiology and the cooling of
physics. The catalog writes each of its systems the same way.

See its [fact sheet](../../../components/systems.md#basesystem)
and its [API entry][skcomplex.core.systems.BaseSystem].

## RunSettings

`RunSettings` names the run settings of a system class: the end time, the time between frames, the solver, the
observables, the initial state, the invariants, the schedule, and the backend. A ready-made system tunes them as class
variables, and `simulate` reads each one only where the call gives no value. An outbreak runs for a month, in frames
ten days apart.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import RunSettings
from skcomplex.core.systems import BaseSystem, System


class OutbreakSystem(BaseSystem):
    until = 30.0
    dt = 10.0

    def __init__(self, label='outbreak', beta=0.3):
        super().__init__(label)
        self.beta = beta

    def _prepare(self, context=None):
        super()._prepare(context)
        infected = Element('infected', values={'x': 10.0})
        growth = lambda s: self.beta * s.target.x * (1 - s.target.x / 1000)
        spread = Interaction(
            'spread',
            roles={'target': 'infected'},
            laws=lambda s: [RateLaw(s.target.x.diff(s.t), growth(s))],
        )
        self.system_ = System(self.label, elements=[infected], interactions=[spread])


outbreak = OutbreakSystem().prepare()
sorted(RunSettings.__annotations__), outbreak.simulate().times.tolist()
```

The run takes the end time 30 and the step 10 from the class, so it has frames at 0, 10, 20, and 30.

A pendulum in mechanics tunes its solver as well, a fixed step of Runge and Kutta of order four, which it passes to
its system.

```python continuation
from skcomplex.core.engine.solvers import RK4Solver


class PendulumSystem(BaseSystem):
    until = 6.0
    dt = 3.0

    def __init__(self, label='pendulum', length=1.0):
        super().__init__(label)
        self.length = length

    def _prepare(self, context=None):
        super()._prepare(context)
        bob = Element('bob', values={'theta': 0.1})
        gravity = Interaction(
            'gravity',
            roles={'target': 'bob'},
            laws=lambda s: [RateLaw(s.target.theta.diff(s.t, 2), -9.81 / self.length * s.target.theta)],
        )
        self.system_ = System(self.label, elements=[bob], interactions=[gravity], solver=RK4Solver(dt=0.01))


pendulum = PendulumSystem().prepare()
type(pendulum.solver_).__name__, pendulum.simulate().get_values('bob.theta').round(3).tolist()
```

The period of a pendulum of one metre is about two seconds, so the angle is -0.1 radians after three seconds, on the
far side, and 0.1 again after six.

**How the engine runs it.** `simulate` reads the settings once per call and uses each one only where the call gives
no value, so an argument of the call wins over the class.

```python continuation
outbreak.simulate(until=10).get_values('infected.x').round(1).tolist()
```

The call asks for 10 days, and the class still gives the step of 10 days, so the run has two frames.

**How it combines.** The settings are the class variables `until`, `dt`, `observables`, `initial_state`,
`invariants`, and `schedule` of a `BaseSystem` subclass, and its arguments `solver` and `backend`.

**Limits and errors.** A class may leave `until` out. Then a call without `until` raises a `ValidationError`.

```python continuation
from skcomplex.core.errors import ValidationError


class OpenOutbreakSystem(OutbreakSystem):
    until = None


try:
    OpenOutbreakSystem().prepare().simulate()
except ValidationError as error:
    message = str(error)
message
```

The message says that the system has no tuned end time and asks to pass `until`. The fix is `simulate(until=30)`, or
the class variable `until = 30.0`.

**A field realization.** A field ties a setting to a parameter by passing it to the call, such as a season of an
epidemic whose length in days is a model parameter.

```python continuation
class SeasonSystem(OutbreakSystem):
    def __init__(self, label='outbreak', beta=0.3, days=60.0):
        super().__init__(label, beta)
        self.days = days


season = SeasonSystem(days=20.0).prepare()
season.simulate(until=season.days).times[-1]
```

See its [API entry][skcomplex.core.components.systems.RunSettings]. It is not a component, so it has no fact sheet.
