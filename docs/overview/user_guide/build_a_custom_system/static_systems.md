# Static systems

Solve a system of equations that has no time axis, such as a dose-response curve, and sweep its parameters. A system
under `StaticTime` has `EquationLaw`s that name their `target`, and a run solves them once and gives one frame.

```python
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import EquationLaw
from skcomplex.core.components.spaces import StaticTime
from skcomplex.core.components.system import System
from skcomplex.core.experiments import Experiment, GridDesign
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


def dose_response(dose: float = 2.0) -> System:
    return System(elements=[DoseElement('cells', dose=dose)], time=StaticTime())


sweep = Experiment(dose_response, GridDesign(), {'dose': [0.0, 2.0]}, outputs=['cells.survival'])
rows = sweep.prepare().run().result_.rows
[round(float(row['cells.survival']), 4) for row in rows]
```

A sweep of a parameter is an `Experiment` with `GridDesign`, one static run per value, as above. The equations are
solved this way:

- The compiler solves the equations with Sage when it finds a unique explicit solution, and `problem.equations` holds
  it.
- Otherwise `RootSolver` solves them with `scipy.optimize.root` and their exact Jacobian, from the initial state as the
  guess. It raises a `SolverError` that names the equations and the largest residual when the solve does not converge.
- A target on a group gives one unknown per member, and reductions such as `total` couple the members.
- The equations of one target from every element and interaction add.
- A system without equations solves to its initial state, so its observables run on quenched states in one frame.

`PassCompiler` reports an equation with no target, or with a target that is not a state, with the code
`equation-target`. A law that does not run under static time keeps its `law-time` issue.

## Log-sum-exp and contact clusters

Reduce over a group without overflow, and measure the clusters that points form when they touch.

```python
from skcomplex.core.engine.runtime import Simulation
from skcomplex.systems import create_system

system = create_system('random_energy_system', n_spins=12, temperature=2.0)
trajectory = Simulation(system).prepare().run(1.0).trajectory_
trajectory.values('ensemble.entropy')[0]
```

`log_sum_exp(expression, over=None)` reduces over the members of a group like `total`, as `log sum_j exp(a_j)`, without
overflow for values of any size. A system-level observable cannot reach a group from another element. So a quantity
built on it is an `EquationLaw` on a scalar element, read by an interaction that binds the group. The system
`random_energy_system` above, Derrida's random energy system, computes its entropy this way.

Quenched disorder is drawn by the factory, so a disorder average sweeps its `random_state` as an axis of `GridDesign`.
The `replicates` of an `Experiment` share one draw of the factory and vary only the dynamics.

`ClusterAnalysis(kind, radius=)` reads the positions of points, joins those closer than twice the radius, and measures
their clusters. A cluster spans along `axis` when it holds points within the radius of both faces of the space.

```python
from skcomplex.core.analysis import ClusterAnalysis

ClusterAnalysis('sizes', radius=0.6).prepare([[0, 0], [1, 0], [5, 5]]).run().sizes_.tolist()
```
