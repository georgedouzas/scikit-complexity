# Kinds of dynamics

Find the laws, elements, spaces, and solvers of the kind of dynamics your system needs, and see how `simulate` turns
the laws into a problem a solver can run.

```python
from skcomplex.core.engine.runtime import simulate
from skcomplex.systems.biology.epidemiology import SIRSystem

system = SIRSystem()
trajectory = simulate(system)
sorted(system.get_tags().laws), sorted(system.get_tags().time), trajectory.values('R.n')[-1].round(1)
```

The SIR model returns rates in continuous time, so its tags choose an ordinary differential equation solver.

Some kinds of dynamics, such as lattice dynamics or operator dynamics, are needed by few systems. None of them is a
package of its own. Its laws, elements, and spaces are parts of the core, and the passes that compile it, its solvers,
and its visuals are parts of the engine, beside those of every other kind.

## Where each kind lives

| Kind | Core | Engine |
| --- | --- | --- |
| agents | `RuleLaw`, `MatchLaw` | the agents passes, `MesaSolver` |
| lattices | `LocalRuleLaw`, `LocalRuleInteraction`, `DriveLaw`, `CascadeLaw`, `ExtremalLaw`, `LatticeElement`, `LatticeSpace`, `ObjectiveSpec`, `SamplerSpec` | `LatticeSolver`, `MonteCarloSolver` |
| finitely generated groups | `CayleySpace` | `CayleyVisual` |
| rewriting | `ProductionLaw`, `SubstitutionLaw`, `WordElement`, `PatchElement` | the rewriting compiler, `RewritingSolver`, `GeometryVisual` |
| operator | `GeneratorLaw`, `DissipatorLaw`, `HilbertSpec` | the operator compiler, `UnitarySolver`, `LindbladSolver`, `JumpTrajectoriesSolver` |
| fields | `FieldElement`, `FluxLaw`, `MeshSpace`, `laplacian` and the other operators | the fields passes, `IMEXEulerSolver`, `CrankNicolsonSolver`, `SSPRKSolver` |
| kinetics | `EncounterLaw`, `moment` | the encounter passes, `KineticSolver` |
| fractional | `StateSpec(order=...)` in `(0, 1)`, `FractionalNoiseLaw` | the fractional passes, `FractionalABMSolver`, `GrunwaldLetnikovSolver`, `FractionalEulerMaruyamaSolver` |
| topology | `TopologyRequestSpec` | the topology passes |
| particles | `PointElement`, `GradientInteraction`, `PairCouplingInteraction` | `VelocityVerletSolver`, `Yoshida4Solver`, `LammpsSolver` |
| control | `ParameterSpec(adaptive=True, bounds=...)` | `PontryaginSolver`, `CollocationSolver` |

`simulate` compiles a system into the problem of its form and runs it. You never name the compiler of a form of the
library: a system of operator elements compiles with the operator compiler, a system of words or patches with the
rewriting compiler, and every other system with the pass compiler, whose passes lower each law to the problem of its
form.

A package of your own can bring a form of its own. Its compiler derives from `BaseCompiler` of
`skcomplex.core.engine.compile`, with passes on `BasePass` and lowerings on `BaseLowering`, and a system passes it to
`simulate` through the `compiler` setting, or tunes it as the class attribute `compiler`. That is the one hook to the
compiled form.

- `prepare(system)` of a compiler validates its settings and the system, and sets `report_`, the issues it found.
- `run(initial_state)` raises the errors of the report, sets `problem_`, the problem, and sets `solver_`, the solver it
  names for the problem.

## Writing a law

A new law derives from `BaseLaw`. It stores its arguments under their own names, and it implements three methods.

- `_get_targets` returns the states it changes, which `prepare` sets as `targets_`.
- `_get_requirements` returns a `RequirementSpec` of the kinds of time and space it runs in.
- `_run` sets `value_`, its value at given values of the symbols it reads, which `prepare` sets as `reads_`.

```python
from typing import Annotated

from sage.all__sagemath_symbolics import var

from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import BaseLaw
from skcomplex.core.components.spaces import ContinuousTime
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RequirementSpec, StateSpec


class DecayLaw(BaseLaw):
    def __init__(self, target, rate, *, label=''):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _get_targets(self):
        return (self.target,)

    @classmethod
    def _get_requirements(cls):
        return RequirementSpec(time=(ContinuousTime,))

    def _run(self, inputs):
        self.value_ = self._evaluate(-self.rate * self.target, inputs)


n, k = var('n k')
law = DecayLaw(n, k).prepare('atom')
law.targets_, sorted(map(str, law.reads_)), law.owner_, law.run({'n': 2.0, 'k': 0.5}).value_
```

The law checks, prepares, and runs on its own. But the passes of the library's compilers are fixed lists of the engine,
so no form takes a law of a new class in a system. `simulate` reports it with the code `laws-mixed`, and a new kind of
dynamics is a change to the engine, or a compiler of your own.

```python continuation
class NucleusElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0, k: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.n = n
        self.k = k

    def _get_laws(self, s):
        return [DecayLaw(s.n, s.k)]


try:
    simulate(System(elements=[NucleusElement('atom')]), until=1)
except ValidationError as error:
    message = str(error)
assert '[laws-mixed]' in message
message
```
