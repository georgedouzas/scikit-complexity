# Stochastic maps

Write a discrete-time stochastic process, such as a random walk or a Markov chain, with update laws that draw random
numbers. The draws come from the simulation's random generator, so two runs with the same `random_state` give the same
trajectory.

## Random draws

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

## Markov chains and hidden Markov sources

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

## Ready-made stochastic maps

The catalog has these systems ready: `persistent_random_walk`, `markov_source`, `hidden_markov_source`, and
`random_walk_with_memory` in stochastic processes, `chaos_game` in fractals, `probabilistic_boolean_network` and
`intrinsic_instability` in biology, and `diluted_hopfield` in neural networks.
