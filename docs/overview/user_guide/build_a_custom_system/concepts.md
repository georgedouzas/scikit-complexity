# Concepts

Learn the few words every custom system is written in. You describe what the system is, and `scikit-complexity` decides
how to solve it.

```python
from skcomplex.core.components.elements import CompartmentElement
from skcomplex.core.components.interactions import RateInteraction
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate

rabbits = CompartmentElement('rabbits', n=10)
growth = RateInteraction('growth', target='rabbits', state='n', rate=lambda s: 0.5 * s.target.n)
system = System(elements=[rabbits], interactions=[growth])
simulate(system, until=2, dt=0.1).values('rabbits.n')[-1].round(2)
```

The element `rabbits` carries the state `n`, the interaction `growth` adds the rate `0.5 n` to it, and the system holds
both. `simulate` checks the system, runs it until the time you give, and returns its trajectory, which holds the value
of `n` at every frame.

An element of your own declares its states as arguments of its `__init__`, each annotated with a spec. `StateSpec`
marks `n` as a state, and the default of the argument is its initial value. `CompartmentElement` is such an element of
the core.

```python continuation
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.specs import StateSpec


class PopulationElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label)
        self.n = n


system = System(elements=[PopulationElement('rabbits', n=10)], interactions=[growth])
simulate(system, until=2, dt=0.1).values('rabbits.n')[-1].round(2)
```

[Defining elements](defining_elements.md) covers states, parameters, roles, and their specs in full.

## Parts

A system is made of **parts**: its elements, its interactions, the laws they return, and the system itself.

- An **element** is an entity that carries state, such as a particle, a population, a compartment, or a market.
- A **group** holds many identical elements of one type as arrays.
- An **interaction** couples elements through **roles** and describes its effect with **laws**.
- A **system** holds elements, interactions, nested systems, an optional space, and an optional time.

These concepts carry no field-specific meaning. A population, a particle, a firm, and a cell are all elements. So a
system in any field of science is a configuration of the same interface, and every ready-made system in
`skcomplex.systems`, such as `SIRSystem`, is a class that builds such a configuration from its model parameters.

## Two main public methods

Every part has two main public methods, and both return the part itself.

- `prepare(...)` validates the part and prepares it against what it works on.
- `run(...)` sets what the part contributes. An element, an interaction, or a system sets its laws in `laws_`. A law
  sets in `value_` its value at given values of the symbols it reads, such as the rate of a `RateLaw`.

A part must be prepared before it runs. Every part also has `get_params`, `set_params`, and `get_tags`, as estimators of
scikit-learn do. `simulate` is a function, since it is what you do to a system and not a part of it.

The interaction `growth` returns a `RateLaw` on `n`. A law has the two main public methods on its own too, so you can check it
before it joins a system. Here it gives its rate at `n = 10`.

```python continuation
from sage.all__sagemath_symbolics import var

from skcomplex.core.components.laws import RateLaw

n = var('n')
RateLaw(n, 0.5 * n).prepare('growth').run({'n': 10.0}).value_
```

## Laws

A **law** says how a state changes. Every law is a class, and these four are the most common.

- `RateLaw` adds a term to the time derivative of a state. Rates on one state add up.
- `NoiseLaw` drives a state with a Wiener process, independent unless laws name a shared or correlated one.
- `JumpLaw` fires a random event that changes states by fixed increments.
- `UpdateLaw` gives the next value of a state after one discrete step.

A **space** gives vector states, such as positions, their dimension. A **time** says whether the system evolves
continuously, in discrete steps, or not at all.

## Runs

A run records **observables**, named values computed from the states at every frame, and checks the **invariants** it
is given.

`skcomplex.core.components` holds the parts. `skcomplex.core.engine` holds `simulate`, the solvers, and the backends.
The laws of a system give its tags, such as its law kinds and its time, and the tags decide the default solver.

```python continuation
sorted(system.get_tags().laws), sorted(system.get_tags().time)
```

| Laws | Form | Default solver |
| --- | --- | --- |
| `RateLaw` | ordinary differential equations | `ScipyODESolver` with `LSODA` |
| `RateLaw` and `NoiseLaw` | stochastic differential equations | `EulerMaruyamaSolver` |
| `JumpLaw` | jump process | `GillespieSolver` |
| `UpdateLaw` | discrete map | `IterationSolver` |
