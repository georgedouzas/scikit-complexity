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
small normal jitter on every step. The interaction `Persistence` sets two
[`UpdateLaw`](../core_components/laws.md#updatelaw) instances, one for the direction `d` and one for the position `x`,
and an [`Element`](../core_components/elements.md#element) with `size=500` holds 500 walkers.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import bernoulli, normal, where
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class Persistence(Interaction):
    def __init__(self, label, *, target: Role, q=0.8):
        super().__init__(label)
        self.target = target
        self.q = q

    def _prepare(self, s):
        super()._prepare(s)
        d, x = s.target.d, s.target.x
        turn = where(bernoulli('turn', 1 - s.q) > 0, -d, d)
        self.laws_ = [UpdateLaw(d, turn), UpdateLaw(x, x + turn + normal('jitter', 0, 0.1))]


walkers = Element('walkers', values={'d': 1.0, 'x': 0.0}, size=500)
system = System(elements=[walkers], interactions=[Persistence('persistence', target='walkers')])
final = system.prepare().simulate(until=200, dt=1.0, random_state=0).get_values('walkers.x')[-1]
round(float(final.var() / 200), 1)
```

After 200 steps the variance of the positions grows by 3.9 per step. A persistent walk whose steps keep their sign
with correlation `r = 2q - 1 = 0.6` spreads at `(1 + r) / (1 - r) = 4` per step for long times, and the sample of 500
walkers comes close to it.

The draws follow these rules.

- A draw takes a fresh value at every step, and every member of a group draws its own value.
- The arguments of a draw can be expressions of states and parameters, and they take the values from before the step.
- Every draw carries a name. Every use of the same name in one interaction reads the same value within a step, and
  draws with different names are independent.
- Two uses of one name with different distributions or arguments fail to compile.
- Within a step, the draws are taken by element path, then draw name, then member index. So the trajectory does not
  depend on the order of the laws.
- The NumPy and Numba [backends](../glossary.md#backend) give equal trajectories for uniform, Bernoulli, and
  categorical draws.

Only update laws and renewal laws take draws. A draw in a rate, noise, jump, constraint, or event law raises
`IncompatibleLawsError`, because noise laws and jump laws give randomness in continuous time. Draws are not states, so
the trajectory does not record them. A system that wants to keep a draw assigns it to a state.

## Markov chains and hidden Markov sources

This section steps a state over a finite alphabet with a transition matrix, and emits an observed symbol from a hidden
state. A Markov chain draws its next value from a row of probabilities that its current value selects. In a hidden
Markov source, the observer sees only a symbol drawn from a row that the hidden value selects. Both are update laws
of a `categorical` draw, whose weights `where` picks from the row of the current value.

Here the weather, `0` for calm and `1` for a storm, stays calm with probability 0.9 and turns calm again from a storm
with probability 0.4. Calm weather emits the symbol `0` more often, and a storm emits `1` more often.

```python
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import UpdateLaw
from skcomplex.core.components.laws.utils import categorical, where
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


def draw_row(name, matrix, value):
    rows = range(len(matrix))
    weights = [sum(where(value == row, matrix[row][column], 0) for row in rows) for column in range(len(matrix[0]))]
    return categorical(name, weights)


class Weather(Interaction):
    def __init__(self, label, *, target: Role, stay=0.9):
        super().__init__(label)
        self.target = target
        self.stay = stay

    def _prepare(self, s):
        super()._prepare(s)
        hidden = draw_row('hidden', [[s.stay, 1 - s.stay], [0.4, 0.6]], s.target.hidden)
        symbol = draw_row('symbol', [[0.7, 0.3], [0.2, 0.8]], hidden)
        self.laws_ = [UpdateLaw(s.target.hidden, hidden), UpdateLaw(s.target.symbol, symbol)]


source = Element('source', values={'hidden': 0, 'symbol': 0})
system = System(elements=[source], interactions=[Weather('weather', target='source')]).prepare()
trajectory = system.simulate(until=10000, dt=1.0, random_state=0)
round(float((trajectory.get_values('source.hidden') == 0).mean()), 1)
```

The weather is calm in a share of 0.8 of the 10000 steps. This is the stationary share of the chain,
`0.4 / (0.1 + 0.4)`.

The two laws follow these rules.

- The target of the chain is a state over the integers from 0 to `n - 1`. Its current value selects the row of the
  transition matrix that the next value is drawn from.
- The symbol is drawn from the row of the emission matrix that the new hidden value selects, so the two sequences
  align frame by frame.
- Both are update laws, so they combine with other update laws.
- Matrix entries are numbers or expressions of parameters, such as `s.stay`.

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
