# Constraints and search

This page lets a system search for a solution step by step. It anneals an objective, optimises it with a population of
candidates, or satisfies logical constraints. Each search is an interaction on a group, so a run records its progress
like any other dynamics.

## Annealing

This section anneals twenty candidates towards the minimum of `(x - 2)^2`. Simulated annealing proposes a random move
and accepts it with a probability that falls as the move worsens the objective and as the search cools. A
[`SamplingInteraction`](../core_components/interactions.md#samplinginteraction) built with `cooling` anneals. Its
`cooling` is a function of the time symbol that multiplies one over the intensity, here `1 / (1 + t)`. The candidates
form a [`GroupElement`](../core_components/elements.md#groupelement) on the sites of a one-dimensional
[`LatticeSpace`](../core_components/spaces.md#latticespace), and a
[`DiscreteTime`](../core_components/spaces.md#discretetime) counts the steps.

```python
from typing import Annotated

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SamplingInteraction
from skcomplex.core.components.spaces import BallNeighborhood, DiscreteTime, LatticeSpace
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import StateSpec


class CandidateElement(Element):
    def __init__(self, label, *, x: Annotated[float, StateSpec()] = 0.0):
        super().__init__(label)
        self.x = x


anneal = SamplingInteraction(
    'anneal',
    target='candidates',
    state='x',
    objective=lambda s: (s.target.x - 2) ** 2,
    move='gaussian',
    cooling=lambda t: 1 / (1 + t),
    neighborhood=BallNeighborhood(metric='manhattan', n_dim=1),
)
candidates = GroupElement('candidates', element=CandidateElement, space=LatticeSpace(20), on='sites')
system = CompositeSystem(elements=[candidates], interactions=[anneal], time=DiscreteTime())
values = simulate(system, until=200, random_state=0).get_values('candidates.x')
round(float(abs(values[0] - 2).mean()), 2), round(float(abs(values[-1] - 2).mean()), 2)
```

The mean distance of the candidates from 2 falls from 2.0 at the start to 0.04 after 200 steps. Each candidate proposes
a Gaussian move, and a Metropolis acceptance keeps it with a probability that falls as the intensity rises with the
cooling. The objective reads no neighbours, so every candidate searches alone, with one interaction per state.

The solver of sampling takes no other laws. So what tracks a run, such as the lowest energy so far, sits in a nested
system, as in `EvolutionaryOptimisersSystem(algorithm='annealing')`.

## Logical constraints

A search that needs a solver of its own is a recipe of the field that uses it. `SatisfiabilityInteraction` reveals
logical constraints step by step to one PySAT solver, which keeps what it learned, as in
`PerceptronSatisfiabilitySystem` of `skcomplex.systems.computer_science.neural_networks`. Each constraint is a clause,
a disjunction of literals, or a cardinality constraint, a bound on how many literals hold. The interaction reports at
every step whether the constraints so far are satisfiable.

## Populations of candidates

This section optimises an objective with a population of candidates that moves through the library pymoo, one
generation per step. `OptimizationInteraction` of `EvolutionaryOptimisersSystem` holds each candidate as a vector state
in the box `bounds`. Here a particle swarm searches for 20 steps.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.computer_science.evolutionary_computation import EvolutionaryOptimisersSystem

system = EvolutionaryOptimisersSystem(algorithm='particle-swarm', random_state=1)
simulate(system, until=20, random_state=1).get_values('search.best')[-1] < 20
```

The block returns `True`, so the best objective after 20 steps lies below 20. The interaction follows these rules.

- The `objective` takes the array of candidates.
- The algorithm is a spec: `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, or
  `CovarianceAdaptationSpec`.
- It reports `best` and `mean` at every step. Once the algorithm stops itself, the candidates stay.
- Runs with the same seed repeat.
