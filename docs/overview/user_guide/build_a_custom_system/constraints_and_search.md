# Constraints and search

This page lets a system search for a solution step by step. It anneals an objective, optimises it with a population of
candidates, or satisfies logical constraints. Each search is an interaction on a group, so a run records its progress
like any other dynamics.

## Annealing

This section anneals twenty candidates towards the minimum of `(x - 2)^2`. Simulated annealing proposes a random move
and accepts it with a probability that falls as the move worsens the objective and as the search cools. An interaction
gives the objective over a temperature that falls with the time, here `T / (1 + t)`, as an `ObjectiveLaw`, and a
[`MetropolisSolver`](../core_components/interactions.md#samplinginteraction) moves the candidates. The candidates form a group on
the sites of a one-dimensional [`LatticeSpace`](../core_components/spaces.md#latticespace), and every sweep is one
step.

```python
import numpy as np

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ObjectiveLaw
from skcomplex.core.components.spaces import LatticeSpace, RadiusNeighborhood
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.systems import System


def cooled(s):
    x = s.target.x
    return [ObjectiveLaw((x - 2) ** 2 / (s.target.T / (1 + s.t)), x)]


candidates = Element('candidates', values={'x': np.zeros(20), 'T': 1.0}, space=LatticeSpace(20))
anneal = Interaction('anneal', roles={'target': 'candidates'}, laws=cooled)
solver = MetropolisSolver(
    moves={'candidates.x': 'gaussian'},
    neighborhood=RadiusNeighborhood(1, metric='manhattan'),
)
system = System(elements=[candidates], interactions=[anneal], solver=solver)
values = system.prepare().simulate(until=200, random_state=0).get_values('candidates.x')
round(float(abs(values[0] - 2).mean()), 2), round(float(abs(values[-1] - 2).mean()), 2)
```

The mean distance of the candidates from 2 falls from 2.0 at the start to 0.04 after 200 steps. Each candidate proposes
a Gaussian move, and a Metropolis acceptance keeps it with a probability that falls as the temperature falls. The
objective reads no neighbours, so every candidate searches alone, and the solver moves every state it names.

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
from skcomplex.systems.computer_science.evolutionary_computation import EvolutionaryOptimisersSystem

system = EvolutionaryOptimisersSystem(algorithm='particle-swarm', random_state=1).prepare()
system.simulate(until=20, random_state=1).get_values('search.best')[-1] < 20
```

The block returns `True`, so the best objective after 20 steps lies below 20. The interaction follows these rules.

- The `objective` takes the array of candidates.
- The algorithm is a spec: `DifferentialEvolutionSpec`, `ParticleSwarmSpec`, `EvolutionStrategySpec`, or
  `CovarianceAdaptationSpec`.
- It reports `best` and `mean` at every step. Once the algorithm stops itself, the candidates stay.
- Runs with the same seed repeat.
