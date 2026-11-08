# Constraints and search

Let a system search for a solution, step by step: satisfy logical constraints, optimise an objective with a population
of candidates, or anneal. Each search is an interaction on a group, so a run records its progress like any other
dynamics.

```python
from typing import Annotated

from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import SatisfiabilityInteraction
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import CardinalitySpec, ClauseSpec, StateSpec


class BitElement(Element):
    def __init__(self, label: str, on: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0) -> None:
        super().__init__(label)
        self.on = on


constraints = [CardinalitySpec((1, 2, 3), 2, 'at-least'), ClauseSpec((-1,)), ClauseSpec((-2,))]
decide = SatisfiabilityInteraction('decide', target='bits', state='on', constraints=constraints)
bits = GroupElement('bits', element=BitElement, size=3)
system = System(elements=[bits], interactions=[decide], time=DiscreteTime())
simulate(system, until=3).values('decide.satisfiable')[1:]
```

`SatisfiabilityInteraction` binds a group of Boolean states and reveals its constraints step by step. Each constraint is
a `ClauseSpec` or a `CardinalitySpec` over literals. A literal is a member index counted from 1, negative for its
negation. One PySAT solver keeps what it learned. It sets the members to a solution when one exists, and reports
`satisfiable` and `revealed` at every step. Above, the steps reveal that at least two of the three bits are on, then
that bit 1 is off, then that bit 2 is off. The first two constraints hold together, and the third makes the set
unsatisfiable.

`OptimizationInteraction` moves a group of candidates through pymoo, one generation per step. Each candidate is a vector
state in the box `bounds`.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.computer_science.evolutionary_computation import EvolutionaryOptimisersSystem

system = EvolutionaryOptimisersSystem(algorithm='particle-swarm', random_state=1)
simulate(system, until=20, random_state=1).values('search.best')[-1] < 20
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
