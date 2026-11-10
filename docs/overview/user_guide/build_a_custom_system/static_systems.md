# Static systems

This page solves a system of equations that has no time axis, such as a dose-response curve, and sweeps its
parameters. A system under a [`StaticTime`](../core_components/spaces.md#statictime) has
[`EquationLaw`](../core_components/laws.md#equationlaw)s that name their `target`, the state each equation solves for.
A run solves them once and gives one frame.

## Equations and sweeps

This section computes the surviving fraction of cells after a dose of radiation, by the linear-quadratic model
`exp(-0.3 D - 0.03 D^2)`, and sweeps the dose. The [`Element`](../core_components/elements.md#element) holds the
survival as a state and the dose as a parameter. A function builds the
[`CompositeSystem`](../core_components/systems.md#compositesystem) from the dose, so the experiment `sweep` can run it
once per value.

```python
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.experiments import sweep
from skcomplex.core.specs import ParameterSpec, StateSpec


class DoseElement(Element):
    def __init__(
        self,
        label: str,
        survival: Annotated[float, StateSpec()] = 0.0,
        dose: Annotated[float, ParameterSpec()] = 2.0,
    ) -> None:
        super().__init__(label)
        self.survival = survival
        self.dose = dose

    def _get_laws(self, s: SimpleNamespace) -> list[EquationLaw]:
        return [EquationLaw(s.survival == exp(-0.3 * s.dose - 0.03 * s.dose**2), target=s.survival)]


def dose_response(dose: float = 2.0) -> CompositeSystem:
    return CompositeSystem(elements=[DoseElement('cells', dose=dose)], time=StaticTime())


result = sweep(dose_response, {'dose': [0.0, 2.0]}, until=0.0, outputs=['cells.survival'])
[round(float(row['cells.survival']), 4) for row in result.rows]
```

The survival is 1.0 without a dose and 0.4868 after a dose of 2. Each static run gives one row of the sweep. The
equations are solved in one of two ways.

- The compiler solves the equations with Sage when it finds a unique explicit solution, and `problem.equations` holds
  it.
- Otherwise `RootSolver` solves them with `scipy.optimize.root` and their exact Jacobian, from the initial state as the
  guess. It raises a `SolverError` that names the equations and the largest residual when the solve does not converge.

Equations combine across the parts of a system.

- A target on a group gives one unknown per member, and reductions such as `total` couple the members.
- The equations of one target from every element and interaction add.
- A system without equations solves to its initial state, so its observables run on quenched states in one frame.

`simulate` raises a `ValidationError` with the code `equation-target` for an equation with no target, or with a target
that is not a state. A law that does not run under static time gives the code `law-time`.

## Log-sum-exp

This section reduces over a group without overflow. `log_sum_exp(expression, over=None)` reduces over the members of a
group like `total`, as `log sum_j exp(a_j)`, without overflow for values of any size. A system-level observable cannot
reach a group from another element. So a quantity built on it is an `EquationLaw` on a scalar element, read by an
interaction that binds the group. The random energy system of Derrida, `RandomEnergySystem` of the catalog, computes
its entropy this way, from the energies of all the configurations of its spins.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.physics.statistical_mechanics import RandomEnergySystem

trajectory = simulate(RandomEnergySystem(n_spins=12, temperature=2.0), until=1.0)
round(float(trajectory.get_values('ensemble.entropy')[0]), 4)
```

With 12 spins at temperature 2.0, the entropy per spin is 0.6309.

Quenched disorder, randomness drawn once and then held fixed, is drawn when the system builds its parts. So a disorder
average sweeps its `random_state` as an axis of `sweep`. The `replicates` of `ensemble` share one draw of the disorder
and vary only the dynamics.

## Contact clusters

This section measures the clusters that points form when they touch. The analysis `clusters(points, kind=...,
radius=...)` reads the positions of points, joins those closer than twice the radius, and measures their clusters. A
cluster spans along `axis` when it holds points within the radius of both faces of the space.

```python
from skcomplex.core.analysis import clusters

clusters([[0, 0], [1, 0], [5, 5]], kind='sizes', radius=0.6).sizes.tolist()
```

The first two points lie one apart, less than twice the radius 0.6, so they form a cluster of size 2. The third point
stands alone.
