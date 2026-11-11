# Systems

A system composes elements, interactions, nested systems, a space, and a time into one model that `simulate` runs.
Elements hold the states, interactions return the laws that change them, a space places the members of a group, and a
time says whether the states flow, step, or are solved once. A `CompositeSystem` takes these components as data. A
`BaseSystem` builds them from the parameters of a model, so users can set, sweep, and calibrate them. `RunSettings`
holds the end time, the step, and the solver that suit a system class. [How the components fit](index.md) shows the
families together, and the [systems page](../../../components/systems.md) compares the system classes and
holds their fact sheets.

## CompositeSystem

A `CompositeSystem` holds the components of a model as they are given. Here prey grow and predators hunt them on every
site of a lattice of three by three sites, and prey move between neighbouring sites. The prey start unevenly, with
more on the first site and fewer on the third.

```python
from skcomplex.core.components.elements import DeclaredElement, GroupElement
from skcomplex.core.components.interactions import CouplingInteraction, RateInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec

animals = {
    'variables': {'prey': DeclaredVariableSpec(1.0, StateSpec()), 'predators': DeclaredVariableSpec(0.5, StateSpec())},
    'prey': [1.5, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}
habitats = GroupElement('habitats', element=DeclaredElement, on='sites', values=animals)
births = RateInteraction(
    'births', target='habitats', state='prey', rate=lambda s: s.target.prey * (1.0 - s.target.predators)
)
hunting = RateInteraction(
    'hunting', target='habitats', state='predators', rate=lambda s: s.target.predators * (s.target.prey - 1.0)
)
dispersal = CouplingInteraction('dispersal', target='habitats', read='prey', write='prey')
ecosystem = CompositeSystem(
    'ecosystem', elements=[habitats], interactions=[births, hunting, dispersal], space=LatticeSpace((3, 3))
)
simulate(ecosystem, until=4, dt=2.0).get_values('habitats.prey').mean(axis=1).round(2).tolist()
```

The mean prey over the lattice rises to 1.65 and falls to 0.53, a cycle of Lotka and Volterra. The group takes the
lattice of the system as its space, since it has none of its own.

The same class holds a savings account. Interest of 5 percent a year grows the balance, and withdrawals of 120 a
year draw it down.

```python continuation
account = DeclaredElement('account', variables={'x': DeclaredVariableSpec(1000.0, StateSpec())})
interest = RateInteraction('interest', target='account', state='x', rate=lambda s: 0.05 * s.target.x)
withdrawals = RateInteraction('withdrawals', target='account', state='x', rate=-120.0)
savings = CompositeSystem('savings', elements=[account], interactions=[interest, withdrawals])
simulate(savings, until=10, dt=5.0).get_values('account.x').round(2).tolist()
```

The withdrawals exceed the interest, so the balance falls to 602.36 after five years and to 91.79 after ten.

**How the engine runs it.** `prepare` prepares the space and the time first, then the elements and the nested
systems on the space, then the interactions. The compiler collects the laws of every component and lowers them to one
problem, and the tags of the system come from its components. The tags choose the default solver.

```python continuation
tags = ecosystem.get_tags()
sorted(tags.laws), sorted(tags.space), sorted(tags.time)
```

The ecosystem holds rate laws on a lattice in continuous time, so `ScipyODESolver` integrates it.

**How it combines.** A system nests other systems through `systems`, and an interaction names an element of a nested
system by its path, such as `savings/account`. Here a family fund grows by 10 percent a year of the balance of the
nested account, a matching gift that the account itself does not pay.

```python continuation
fund = DeclaredElement('fund', variables={'x': DeclaredVariableSpec(0.0, StateSpec())})
deposit = RateInteraction(
    'deposit', target='fund', state='x', rate=lambda s: 0.1 * s.sources[0].x, sources=['savings/account']
)
family = CompositeSystem('family', elements=[fund], systems=[savings], interactions=[deposit])
trajectory = simulate(family, until=2, dt=1.0)
trajectory.get_values('savings/account.x').round(2).tolist(), trajectory.get_values('fund.x').round(2).tolist()
```

The account falls to 928.22 and 852.76 under its own laws, and the fund grows to 96.44 and 185.52. A nested system
may also be a `BaseSystem` subclass, which the composite replaces by the components it builds.

**Limits and errors.** The components are fixed when the system is built, so a parameter sweep over a composite rebuilds
the components by hand. A `BaseSystem` builds them from parameters. Every role must name an element of the system, and
the run raises a `RoleError`, a kind of `ValidationError`, for a path that names none.

```python continuation
from skcomplex.core.errors import ValidationError

typo = RateInteraction('interest', target='acount', state='x', rate=lambda s: 0.05 * s.target.x)
try:
    simulate(CompositeSystem(elements=[account], interactions=[typo]), until=1)
except ValidationError as error:
    message = str(error)
message
```

The message says that the interaction `interest` binds the role `target` to `acount`, which is not an element or
group path of the system. The fix is the label of the element, `account`.

**A field realization.** The catalog writes a system as a `BaseSystem` subclass whose hook `_build_system` returns the
composite. Here the savings account takes its rate and its withdrawals as model parameters.

```python continuation
from skcomplex.core.components.systems import BaseSystem


class SavingsSystem(BaseSystem):
    def __init__(self, label='savings', rate=0.05, withdrawal=120.0):
        super().__init__(label)
        self.rate = rate
        self.withdrawal = withdrawal

    def _build_system(self):
        account = DeclaredElement('account', variables={'x': DeclaredVariableSpec(1000.0, StateSpec())})
        interest = RateInteraction('interest', target='account', state='x', rate=lambda s: self.rate * s.target.x)
        withdrawals = RateInteraction('withdrawals', target='account', state='x', rate=-self.withdrawal)
        return CompositeSystem(self.label, elements=[account], interactions=[interest, withdrawals])


simulate(SavingsSystem(withdrawal=50.0), until=10, dt=10.0).get_values('account.x').round(2).tolist()
```

Withdrawals of 50 a year equal the interest on 1000 at 5 percent, so the balance stays at 1000.

See its [fact sheet](../../../components/systems.md#compositesystem)
and its [API entry][skcomplex.core.components.systems.CompositeSystem].

## BaseSystem

A `BaseSystem` is the base of a system class with model parameters. The subclass takes the parameters as arguments of
its `__init__` and stores each under the same name, as a scikit-learn estimator does, and it builds its components in
the hook `_build_system`. Class variables such as `until` and `dt` are its tuned run settings. Here an outbreak grows
logistically in a town, at the rate `beta` among `population` people.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import BaseSystem, CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec


class OutbreakSystem(BaseSystem):
    until = 30.0
    dt = 10.0

    def __init__(self, label='outbreak', beta=0.3, population=1000.0):
        super().__init__(label)
        self.beta = beta
        self.population = population

    def _build_system(self):
        infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
        growth = lambda s: self.beta * s.target.x * (1 - s.target.x / self.population)
        spread = RateInteraction('spread', target='infected', state='x', rate=growth)
        return CompositeSystem(self.label, elements=[infected], interactions=[spread])


outbreak = OutbreakSystem()
outbreak.get_params(), simulate(outbreak).get_values('infected.x').round(1).tolist()
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

    def _build_system(self):
        cup = DeclaredElement('coffee', variables={'T': DeclaredVariableSpec(90.0, StateSpec())})
        newton = lambda s: -self.k * (s.target.T - self.room)
        cooling = RateInteraction('cooling', target='coffee', state='T', rate=newton)
        return CompositeSystem(self.label, elements=[cup], interactions=[cooling])


cup = CoolingSystem()
before = simulate(cup).get_values('coffee.T').round(1).tolist()
after = simulate(cup.set_params(k=0.05)).get_values('coffee.T').round(1).tolist()
before, after
```

With `k = 0.1` the coffee falls from 90 to 45.8 and 29.5 degrees, and with `k = 0.05` it cools more slowly, to 62.5 and
45.8 degrees.

**How the engine runs it.** `prepare` builds the components again when a parameter has changed by value since the last
build, and prepares them as `system_`. The tags of the system are those of its components, never declared, and `clone`
copies the parameters, as for any scikit-learn estimator.

```python continuation
from sklearn.base import clone

copy = clone(cup)
copy.get_params(), sorted(copy.get_tags().laws), type(copy.prepare().system_).__name__
```

The clone has the cooling constant 0.05 that `set_params` gave the cup, its laws are rates, and its components form a
`CompositeSystem`.

**How it combines.** A system class nests inside a `CompositeSystem` through `systems`, and its states are named by
its path, such as `outbreak/infected.x`. The parameters are what calibration fits and what experiments sweep.

```python continuation
town = CompositeSystem('town', systems=[OutbreakSystem()])
simulate(town, until=10, dt=10.0).get_values('outbreak/infected.x').round(1).tolist()
```

The nested outbreak reaches 168.7 infected after ten days, as it does alone.

**Limits and errors.** The hook `_build_system` returns the `CompositeSystem` of the components, and `prepare` raises a
`ValidationError` for anything else, such as one element.

```python continuation
from skcomplex.core.errors import ValidationError


class TankSystem(BaseSystem):
    def _build_system(self):
        return DeclaredElement('tank', variables={'x': DeclaredVariableSpec(1.0, StateSpec())})


try:
    TankSystem().prepare()
except ValidationError as error:
    message = str(error)
message
```

The message reads "TankSystem builds its components as a CompositeSystem, not as DeclaredElement." The fix is to return
`CompositeSystem(self.label, elements=[tank])`.

**A field realization.** The subclasses above are field realizations, an outbreak of epidemiology and the cooling of
physics. The catalog writes each of its systems the same way.

See its [fact sheet](../../../components/systems.md#basesystem)
and its [API entry][skcomplex.core.components.systems.BaseSystem].

## RunSettings

`RunSettings` is the mapping of the tuned run settings of a system class, which `get_run_settings` returns and
`simulate` reads for every argument a call leaves out. An outbreak runs for a month, in frames ten days apart.

```python
from skcomplex.core.components.elements import DeclaredElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import BaseSystem, CompositeSystem, RunSettings
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import DeclaredVariableSpec, StateSpec


class OutbreakSystem(BaseSystem):
    until = 30.0
    dt = 10.0

    def __init__(self, label='outbreak', beta=0.3):
        super().__init__(label)
        self.beta = beta

    def _build_system(self):
        infected = DeclaredElement('infected', variables={'x': DeclaredVariableSpec(10.0, StateSpec())})
        growth = lambda s: self.beta * s.target.x * (1 - s.target.x / 1000)
        spread = RateInteraction('spread', target='infected', state='x', rate=growth)
        return CompositeSystem(self.label, elements=[infected], interactions=[spread])


settings = OutbreakSystem().get_run_settings()
settings['until'], settings['dt'], sorted(settings) == sorted(RunSettings.__annotations__)
```

The settings hold the end time 30 and the step 10, and every key that `RunSettings` names. Settings that the class
leaves out are `None`, or an empty sequence for `observables`.

A pendulum in mechanics tunes its solver as well, a fixed step of Runge and Kutta of order four.

```python continuation
from skcomplex.core.engine.solvers import RK4Solver


class PendulumSystem(BaseSystem):
    until = 6.0
    dt = 3.0
    solver = RK4Solver(dt=0.01)

    def __init__(self, label='pendulum', length=1.0):
        super().__init__(label)
        self.length = length

    def _build_system(self):
        variables = {'theta': DeclaredVariableSpec(0.1, StateSpec()), 'omega': DeclaredVariableSpec(0.0, StateSpec())}
        bob = DeclaredElement('bob', variables=variables)
        swing = RateInteraction('swing', target='bob', state='theta', rate=lambda s: s.target.omega)
        gravity = RateInteraction(
            'gravity', target='bob', state='omega', rate=lambda s: -9.81 / self.length * s.target.theta
        )
        return CompositeSystem(self.label, elements=[bob], interactions=[swing, gravity])


pendulum = PendulumSystem()
type(pendulum.get_run_settings()['solver']).__name__, simulate(pendulum).get_values('bob.theta').round(3).tolist()
```

The period of a pendulum of one metre is about two seconds, so the angle is -0.1 radians after three seconds, on the
far side, and 0.1 again after six.

**How the engine runs it.** `simulate` reads the settings once per call and uses each one only where the call gives
no value, so an argument of the call wins over the class.

```python continuation
outbreak = OutbreakSystem()
simulate(outbreak, until=10).get_values('infected.x').round(1).tolist()
```

The call asks for 10 days, and the class still gives the step of 10 days, so the run has two frames.

**How it combines.** The settings are the class variables `until`, `dt`, `solver`, `observables`, `initial_state`,
`invariants`, `schedule`, `compiler`, and `backend` of a `BaseSystem` subclass. A system whose settings depend on its
parameters overrides `get_run_settings`.

**Limits and errors.** A `CompositeSystem` has no tuned settings, and a class may leave `until` out. Then a call
without `until` makes `simulate` raise a `ValidationError`.

```python continuation
from skcomplex.core.errors import ValidationError


class OpenOutbreakSystem(OutbreakSystem):
    until = None


try:
    simulate(OpenOutbreakSystem())
except ValidationError as error:
    message = str(error)
message
```

The message reads "OpenOutbreakSystem has no tuned end time. Pass `until`." The fix is `simulate(system, until=30)`,
or the class variable `until = 30.0`.

**A field realization.** A field ties a setting to a parameter by overriding `get_run_settings`, such as a season of
an epidemic whose length in days is a model parameter.

```python continuation
class SeasonSystem(OutbreakSystem):
    def __init__(self, label='outbreak', beta=0.3, days=60.0):
        super().__init__(label, beta)
        self.days = days

    def get_run_settings(self):
        return super().get_run_settings() | {'until': self.days}


SeasonSystem(days=20.0).get_run_settings()['until']
```

See its [API entry][skcomplex.core.components.systems.RunSettings]. It is not a component, so it has no fact sheet.
