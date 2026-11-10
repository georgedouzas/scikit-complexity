# Stochastic maps

This page writes discrete-time stochastic processes, such as a random walk or a Markov chain, with update laws that
draw random numbers. An update law gives the next value of a state from the values of the current step. The draws come
from the random generator of the run, so two runs with the same `random_state` give the same
[trajectory](../glossary.md#trajectory).

## Random draws

This section puts random draws inside update laws and runs a group of random walkers. A draw is a symbolic random
value that enters a law like any other expression. `uniform(name, low, high)`, `normal(name, mean, deviation)`,
`bernoulli(name, p)`, and `categorical(name, weights)` build one.

Here each walker of a persistent random walk keeps its direction with probability `q` and turns back otherwise, with a
small normal jitter on every step. An [`Element`](../core_components/elements.md#element) subclass returns two
[`UpdateLaw`](../core_components/laws.md#updatelaw) instances, one for the direction `d` and one for the position `x`,
and a [`GroupElement`](../core_components/elements.md#groupelement) makes 500 walkers of it inside a
[`CompositeSystem`](../core_components/systems.md#compositesystem).

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, normal, where
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
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


walkers = CompositeSystem(elements=[GroupElement('walkers', element=PersistentElement, size=500)])
final = simulate(walkers, until=200, dt=1.0, random_state=0).get_values('walkers.x')[-1]
round(float(final.var() / 200), 1)
```

After 200 steps the variance of the positions grows by 3.9 per step. A persistent walk whose steps keep their sign
with correlation `r = 2q - 1 = 0.6` spreads at `(1 + r) / (1 - r) = 4` per step for long times, and the sample of 500
walkers comes close to it.

The draws follow these rules.

- A draw takes a fresh value at every step, and every member of a group draws its own value.
- The arguments of a draw can be expressions of states and parameters, and they take the values from before the step.
- Every draw carries a name. Every use of the same name in one element reads the same value within a step, and draws
  with different names are independent.
- Two uses of one name with different distributions or arguments fail to compile.
- Within a step, the draws are taken by element path, then draw name, then member index. So the trajectory does not
  depend on the order of the laws.
- The NumPy and Numba [backends](../glossary.md#backend) give equal trajectories for uniform, Bernoulli, and
  categorical draws.

Only update laws and renewal laws take draws. A draw in a rate, noise, jump, boundary, equation, or crossing law raises
`IncompatibleLawsError`, because noise laws and jump laws give randomness in continuous time. Draws are not states, so
the trajectory does not record them. A system that wants to keep a draw assigns it to a state.

## Markov chains and hidden Markov sources

This section steps a state over a finite alphabet with a transition matrix, and emits an observed symbol from a hidden
state. A Markov chain draws its next value from a row of probabilities that its current value selects. In a hidden
Markov source, the observer sees only a symbol drawn from a row that the hidden value selects.
[`TransitionLaw`](../core_components/laws.md#transitionlaw)`(targets, matrix)` is the step of a discrete-time Markov
chain, and `TransitionLaw(targets, matrix, source=source)` draws an observed state from the row that a hidden state
selects.

Here the weather stays calm with probability 0.9 and turns calm again from a storm with probability 0.4. Calm weather
emits the symbol `'a'` more often, and a storm emits `'b'` more often.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import TransitionLaw
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
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
            TransitionLaw(s.symbol, [[0.7, 0.3], [0.2, 0.8]], source=s.hidden),
        ]


trajectory = simulate(CompositeSystem(elements=[SourceElement('source')]), until=10000, dt=1.0, random_state=0)
round(float((trajectory.get_values('source.hidden') == 'calm').mean()), 1)
```

The weather is calm in a share of 0.8 of the 10000 steps. This is the stationary share of the chain,
`0.4 / (0.1 + 0.4)`.

The two laws follow these rules.

- The target of a `TransitionLaw` is a state over a finite alphabet, a `FiniteEnumeratedSet` of labels or the integers
  from 0 to `n - 1`. Its current value selects the row of the transition matrix that the next value is drawn from.
- A `TransitionLaw` with a `source` draws the observed state from the row of an emission matrix that the hidden state
  selects. It reads the hidden state after the same step, so the two sequences align frame by frame. Its matrix has one
  row per hidden value and one column per observed value, and it is symbolic only.
- Both are update laws, so they combine with other update laws.
- Matrix entries are numbers or expressions of parameters. `simulate` checks that the entries are nonnegative and that
  every row sums to 1 at the initial parameter values, and raises an error that names the entry otherwise.

A probability of staying above 1 breaks the first of these checks.

```python continuation
from skcomplex.core.errors import ValidationError

try:
    simulate(CompositeSystem(elements=[SourceElement('source', stay=1.5)]), until=10)
except ValidationError as error:
    message = str(error)
assert 'below 0' in message
message
```

The message says that the entry (0, 1) of the transition law is -0.5, below 0, and asks for a matrix with nonnegative
entries whose rows sum to 1. The fix is a value of `stay` between 0 and 1.

## Ready-made stochastic maps

This section points to the catalog systems that are stochastic maps, so you can start from one of them. They are
`PersistentRandomWalkSystem`, `MarkovSourceSystem`, `HiddenMarkovSourceSystem`, and `RandomWalkWithMemorySystem` of
`skcomplex.systems.mathematics.stochastic_processes`, `ChaosGameSystem` of `skcomplex.systems.mathematics.fractals`,
`ProbabilisticBooleanNetworkSystem` of `skcomplex.systems.biology.systems_biology`, `IntrinsicInstabilitySystem` of
`skcomplex.systems.biology.ecology`, and `DilutedHopfieldSystem` of
`skcomplex.systems.computer_science.neural_networks`.

```python continuation
from skcomplex.systems.mathematics.stochastic_processes import MarkovSourceSystem

MarkovSourceSystem().get_tags().randomness
```

The tag `randomness` of `MarkovSourceSystem` reads `'stochastic'`.
