# Constraints and search

Let a system search for a solution, step by step: anneal an objective, optimise it with a population of candidates, or
satisfy logical constraints. Each search is an interaction on a group, so a run records its progress like any other
dynamics.

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

Twenty candidates anneal towards the minimum of `(x - 2)^2`. Each proposes a Gaussian move, and a Metropolis
acceptance keeps it with a probability that falls as the intensity rises with the cooling. The objective reads no
neighbours, so every candidate searches alone.

A search that needs a solver of its own is a recipe of the field that uses it. `SatisfiabilityInteraction` reveals
logical constraints step by step to one PySAT solver, which keeps what it learned, as in
`PerceptronSatisfiabilitySystem` of `skcomplex.systems.computer_science.neural_networks`. Each constraint is a clause
or a cardinality over literals, and the interaction reports at every step whether the constraints so far are
satisfiable.

`OptimizationInteraction` of `EvolutionaryOptimisersSystem` moves a group of candidates through pymoo, one generation per
step. Each candidate is a vector state in the box `bounds`.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.computer_science.evolutionary_computation import EvolutionaryOptimisersSystem

system = EvolutionaryOptimisersSystem(algorithm='particle-swarm', random_state=1)
simulate(system, until=20, random_state=1).get_values('search.best')[-1] < 20
```

- The `objective` takes the array of candidates.
- The algorithm is a spec: `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, or
  `CovarianceAdaptationSpec`.
- It reports `best` and `mean` at every step. Once the algorithm stops itself, the candidates stay.
- Runs with the same seed repeat.

`SamplingInteraction(cooling=)` anneals. Its `cooling` is a function of the time symbol that multiplies one over the
intensity. So a Metropolis acceptance with the `'gaussian'` move anneals a group of candidates on a one-dimensional
`LatticeSpace` whose objective reads no neighbours, with one interaction per state. The solver of sampling takes no
other laws. So what tracks a run, such as the lowest energy so far, sits in a nested system, as in
`EvolutionaryOptimisersSystem(algorithm='annealing')`.
