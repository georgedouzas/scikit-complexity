# Kinds of dynamics

Find the laws, elements, spaces, and solvers of the kind of dynamics your system needs, and see how a compiler turns
the laws into a problem a solver can run.

```python
from skcomplex.core.engine.compile import BaseCompiler, PassCompiler
from skcomplex.systems import create_system

compiler = PassCompiler().prepare(create_system('sir')).run()
isinstance(compiler, BaseCompiler), compiler.report_.ok, compiler.problem_.form.value
```

Some kinds of dynamics, such as lattice dynamics or operator dynamics, are needed by few systems. None of them is a
package of its own. Its laws, elements, and spaces are parts of the core, and the passes that compile it, its solvers,
and its visuals are parts of the engine, beside those of every other kind.

## Where each kind lives

| Kind | Core | Engine |
| --- | --- | --- |
| agents | `RuleLaw`, `MatchLaw` | the agents passes, `MesaSolver` |
| lattices | `LocalRuleLaw`, `LocalRuleInteraction`, `DriveLaw`, `CascadeLaw`, `ExtremalLaw`, `LatticeElement`, `LatticeSpace`, `ObjectiveSpec`, `SamplerSpec` | `LatticeSolver`, `MonteCarloSolver` |
| finitely generated groups | `CayleySpace` | `CayleyVisual` |
| rewriting | `ProductionLaw`, `SubstitutionLaw`, `WordElement`, `PatchElement` | `RewritingCompiler`, `RewritingSolver`, `GeometryVisual` |
| operator | `GeneratorLaw`, `DissipatorLaw`, `HilbertSpec` | a system compiler, `UnitarySolver`, `LindbladSolver`, `JumpTrajectoriesSolver` |
| fields | `FieldElement`, `FluxLaw`, `MeshSpace`, `laplacian` and the other operators | the fields passes, `IMEXEulerSolver`, `CrankNicolsonSolver`, `SSPRKSolver` |
| kinetics | `EncounterLaw`, `moment` | the encounter passes, `KineticSolver` |
| fractional | `StateSpec(order=...)` in `(0, 1)`, `FractionalNoiseLaw` | the fractional passes, `FractionalABMSolver`, `GrunwaldLetnikovSolver`, `FractionalEulerMaruyamaSolver` |
| topology | `TopologyRequestSpec` | the topology passes |
| particles | `PointElement`, `GradientInteraction`, `PairCouplingInteraction` | `VelocityVerletSolver`, `Yoshida4Solver`, `LammpsSolver` |
| control | `ParameterSpec(adaptive=True, bounds=...)` | `PontryaginSolver`, `CollocationSolver` |

Every compiler derives from `BaseCompiler` and speaks the two verbs of every part.

- `prepare(system)` validates its settings and the system, and sets `report_`, the issues it found.
- `run(initial_state)` raises the errors of the report, sets `problem_`, the problem, and sets `solver_`, the solver it
  names for the problem.

`Simulation` takes the compiler as the `compiler` setting, and a compiler alone compiles a system with
`prepare(system).run().problem_`. The default `PassCompiler` compiles every system but operator ones through its passes,
which lower each law to the problem of its form. For a system that needs another compiler, it names that compiler.

- An operator system takes `compiler=OperatorCompiler()`.
- A rewriting system takes `compiler=RewritingCompiler()`.
- A package of your own can bring the compiler of a form of its own, with the solver it names.

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
from skcomplex.core.components.system import System
from skcomplex.core.engine.compile import PassCompiler
from skcomplex.core.specs import ParameterSpec, RequirementSpec, StateSpec


class DecayLaw(BaseLaw):
    def __init__(self, target, rate, *, label=''):
        super().__init__(label)
        self.target = target
        self.rate = rate

    def _check(self):
        pass

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

The law checks, prepares, and runs on its own. But the passes and the system compilers are fixed lists of the engine,
so no form takes a law of a new class in a system. The compiler reports it with the code `laws-mixed`, and a new kind of
dynamics is a change to the engine.

```python continuation
class NucleusElement(Element):
    def __init__(self, label, *, n: Annotated[float, StateSpec()] = 1.0, k: Annotated[float, ParameterSpec()] = 1.0):
        super().__init__(label)
        self.n = n
        self.k = k

    def _get_laws(self, s):
        return [DecayLaw(s.n, s.k)]


report = PassCompiler().prepare(System(elements=[NucleusElement('atom')])).report_
[issue.code for issue in report.errors]
```
