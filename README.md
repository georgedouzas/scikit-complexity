[black badge]: <https://img.shields.io/badge/%20style-black-000000.svg>
[black]: <https://github.com/psf/black>
[docformatter badge]: <https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg>
[docformatter]: <https://github.com/PyCQA/docformatter>
[ruff badge]: <https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json>
[ruff]: <https://github.com/charliermarsh/ruff>
[mypy badge]: <http://www.mypy-lang.org/static/mypy_badge.svg>
[mypy]: <http://mypy-lang.org>
[properdocs badge]: <https://img.shields.io/badge/docs-properdocs-blue.svg?style=flat>
[properdocs]: <https://properdocs.org>
[pip-audit badge]: <https://img.shields.io/badge/security-pip--audit-green>
[pip-audit]: <https://github.com/pypa/pip-audit>
[bandit badge]: <https://img.shields.io/badge/security-bandit-yellow>
[bandit]: <https://github.com/PyCQA/bandit>
[pytest badge]: <https://img.shields.io/badge/tests-pytest-blue>
[pytest]: <https://github.com/pytest-dev/pytest>
[coverage badge]: <https://img.shields.io/badge/coverage-pytest--cov-blue>
[coverage]: <https://github.com/nedbat/coveragepy>
[interrogate badge]: <https://img.shields.io/badge/docstring-interrogate-blue>
[interrogate]: <https://github.com/econchick/interrogate>
[pre-commit badge]: <https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit>
[pre-commit]: <https://github.com/pre-commit/pre-commit>
[nox badge]: <https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg>
[nox]: <https://github.com/wntrblm/nox>
[version badge]: <https://img.shields.io/pypi/v/scikit-complexity.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/scikit-complexity.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/scikit-complexity>
[discussions]: <https://github.com/georgedouzas/scikit-complexity/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/georgedouzas/scikit-complexity>
[ci]: <https://github.com/georgedouzas/scikit-complexity/actions?query=workflow>
[ci badge]: <https://github.com/georgedouzas/scikit-complexity/actions/workflows/ci.yml/badge.svg?branch=main>
[doc]: <https://github.com/georgedouzas/scikit-complexity/actions?query=workflow>
[doc badge]: <https://github.com/georgedouzas/scikit-complexity/actions/workflows/doc.yml/badge.svg?branch=main>

# scikit-complexity

| Category          | Tools    |
| ------------------| -------- |
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Testing**       | [![pytest][pytest badge]][pytest] [![coverage][coverage badge]][coverage] [![interrogate][interrogate badge]][interrogate] |
| **Security**      | [![pip-audit][pip-audit badge]][pip-audit] [![bandit][bandit badge]][bandit] |
| **Automation**    | [![nox][nox badge]][nox] [![pre-commit][pre-commit badge]][pre-commit] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![properdocs][properdocs badge]][properdocs]|
| **Communication** | [![discussions][discussions badge]][discussions] |

## Overview

With `scikit-complexity`, you can describe a system in any field of science with just a few lines of code, and it takes
care of the rest: it works out the mathematics, picks a solver unless you choose one, runs the simulation, and can
stream it live to the [Rerun](https://rerun.io) viewer.

Complex systems are usually modelled in separate tools with separate conventions, often one for each field.
Here they all share one small vocabulary, in the spirit of [scikit-learn](https://scikit-learn.org): elements carry
state, interactions change it through laws, and systems nest and compose. Learn it once, and every field opens up.

<table>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/cooling_coffee.png" alt="Cooling coffee" width="100%"><br><b>Physics</b><br><sub>Cooling coffee</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/duffing.png" alt="Duffing" width="100%"><br><b>Physics</b><br><sub>Duffing</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/mirollo_strogatz.png" alt="Mirollo-Strogatz" width="100%"><br><b>Physics</b><br><sub>Mirollo-Strogatz</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/brusselator_ode.png" alt="Brusselator" width="100%"><br><b>Chemistry</b><br><sub>Brusselator</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/cubic_autocatalysis.png" alt="Cubic autocatalysis" width="100%"><br><b>Chemistry</b><br><sub>Cubic autocatalysis</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/cyclic_cellular_automaton.png" alt="Cyclic cellular automaton" width="100%"><br><b>Chemistry</b><br><sub>Cyclic cellular automaton</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/agent_based_sir.png" alt="Agent-based SIR" width="100%"><br><b>Biology</b><br><sub>Agent-based SIR</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/boolean_repressilator.png" alt="Boolean repressilator" width="100%"><br><b>Biology</b><br><sub>Boolean repressilator</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/replicator_dynamics.png" alt="Replicator dynamics" width="100%"><br><b>Biology</b><br><sub>Replicator dynamics</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/correlated_geometric_brownian_motion.png" alt="Correlated geometric Brownian motions" width="100%"><br><b>Economics</b><br><sub>Correlated geometric Brownian motions</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/fishery_capital.png" alt="Fishery capital" width="100%"><br><b>Economics</b><br><sub>Fishery capital</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/geometric_brownian_motion.png" alt="Geometric Brownian motion" width="100%"><br><b>Economics</b><br><sub>Geometric Brownian motion</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/chua.png" alt="Chua's circuit" width="100%"><br><b>Engineering</b><br><sub>Chua's circuit</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/memristor_oscillator.png" alt="Memristor oscillator" width="100%"><br><b>Engineering</b><br><sub>Memristor oscillator</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/tunnel_diode_oscillator.png" alt="Tunnel-diode oscillator" width="100%"><br><b>Engineering</b><br><sub>Tunnel-diode oscillator</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/finite_time_blow_up.png" alt="Finite-time blow-up" width="100%"><br><b>Mathematics</b><br><sub>Finite-time blow-up</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/hypergraph_random_walk.png" alt="Random walk on a hypergraph" width="100%"><br><b>Mathematics</b><br><sub>Random walk on a hypergraph</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/julia.png" alt="Julia set" width="100%"><br><b>Mathematics</b><br><sub>Julia set</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/baumslag_solitar_automaton.png" alt="Majority automaton on the Baumslag-Solitar group" width="100%"><br><b>Computer science</b><br><sub>Majority automaton on the Baumslag-Solitar group</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/free_group_automaton.png" alt="Majority automaton on the free group" width="100%"><br><b>Computer science</b><br><sub>Majority automaton on the free group</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/game_of_life.png" alt="Game of Life" width="100%"><br><b>Computer science</b><br><sub>Game of Life</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/hegselmann_krause.png" alt="Hegselmann-Krause" width="100%"><br><b>Social science</b><br><sub>Hegselmann-Krause</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/hypergraph_public_goods.png" alt="Public goods game on a hypergraph" width="100%"><br><b>Social science</b><br><sub>Public goods game on a hypergraph</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/prisoners_dilemma_tournament.png" alt="Prisoner's dilemma tournament" width="100%"><br><b>Social science</b><br><sub>Prisoner's dilemma tournament</sub></td>
  </tr>
</table>

<p align="center"><em>A few of the ready-to-run systems, each with a documented example in the
<a href="https://georgedouzas.github.io/scikit-complexity">gallery</a>.</em></p>

## Features

- Every kind of dynamics, one interface. Differential equations, stochastic processes, jump processes, discrete maps,
  cellular automata, agents, networks and hypergraphs, fields and PDEs, quantum systems, and systems that learn by gradient
  descent.
- The form of your system is inferred. Write the laws, and the framework decides whether it is an ODE, an SDE or a jump
  process, and compiles it for NumPy, Numba, or JAX.
- Symbols when you want them, code when you don't. Laws are Sage symbolic expressions you can inspect and solve exactly,
  or plain Python functions.
- Watch it run. Stream any simulation to Rerun in real time, and change its parameters while it runs.
- Analysis included. Nearly 50 analyses, from Lyapunov exponents and entropy to network motifs, scaling laws and system
  identification, plus experiments for parameter sweeps, sensitivity, calibration and basin stability.
- Standing on mature libraries. SciPy, NetworkX, Mesa, LAMMPS, QuTiP, JAX and others do the heavy lifting.
- Add your own parts without changing the library. Its systems use the same public classes you do, so each one doubles as
  an example for your own.

## Installation

`scikit-complexity` supports Python 3.12, 3.13, and 3.14 on Linux and on macOS 13 or later with Apple silicon, and is
available on PyPI. On Python 3.14 the install builds `memory-allocator` from source, so it needs a C compiler:

```bash
pip install scikit-complexity
```

The installation includes Sage through the [passagemath](https://github.com/passagemath/passagemath) distributions,
LAMMPS, Mesa, Numba, JAX, and Rerun. Windows has no passagemath wheels, so on Windows the package runs in WSL.

For development, clone the repository and install the project with [PDM](https://github.com/pdm-project/pdm):

```bash
git clone https://github.com/georgedouzas/scikit-complexity.git
cd scikit-complexity
pdm install
```

## Usage

The library speaks scikit-learn. What you define is an estimator, a class with `get_params`, `set_params`, and
`clone`: elements, interactions, and systems. What you do to them is a function: `simulate` runs a system as
`cross_validate` runs a model, analyses measure the run as `sklearn.metrics` scores one, and experiments run many as
the model selection functions do.

There are two ways in. Select a ready-made system from the catalog, or build a custom system from the general classes
of `skcomplex.core.components`.

### Select a ready-made system

Every ready-made system is a class with a field, topics, references, and parameters. You import it from its subfield,
the package `skcomplex.systems.<field>.<subfield>`, and `all_systems` finds them, as `all_estimators` does in
scikit-learn.

List the systems of a field:

```python
from skcomplex.systems import all_systems

ecology = [name for name, _ in all_systems(field='biology/ecology')]
```

Search every field by topic, or by the capability tags of a system:

```python continuation
oscillators = [name for name, _ in all_systems(topics='oscillation')]
stochastic = [name for name, _ in all_systems(field='chemistry', tags={'randomness': 'stochastic'})]
```

Import the class you chose from its subfield and read what it is:

```python continuation
from skcomplex.systems.biology.ecology import LotkaVolterraSystem

LotkaVolterraSystem.topics, LotkaVolterraSystem.references, LotkaVolterraSystem().get_params()
```

Simulate it with its tuned settings, or with parameters and settings of your own:

```python continuation
from skcomplex.core.engine.runtime import simulate

trajectory = simulate(LotkaVolterraSystem())
fast_prey = simulate(LotkaVolterraSystem(r=1.5), until=50, dt=0.1)
round(float(fast_prey.values('prey.n').max()))
```

Each run can stream to the Rerun viewer:

<table>
  <tr>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/lotka_volterra.png" alt="Prey and predators cycle" width="100%"><br><code>lotka_volterra</code><br><sub>Prey and predators cycle</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/sir.png" alt="An epidemic rises and fades" width="100%"><br><code>sir</code><br><sub>An epidemic rises and fades</sub></td>
    <td align="center" valign="top" width="33%"><img src="https://raw.githubusercontent.com/georgedouzas/scikit-complexity/main/docs/assets/readme/wolf_sheep.png" alt="Wolves, sheep and grass on a grid" width="100%"><br><code>wolf_sheep</code><br><sub>Wolves, sheep and grass on a grid</sub></td>
  </tr>
</table>

The [catalog](https://georgedouzas.github.io/scikit-complexity/systems/catalog/) lists every system by field and topic,
and the page of each system shows the classes it is built from.

### Build a custom system

A custom system needs only the core. An element declares its states, and an interaction declares its roles and returns
its laws, where the namespace `s` holds the symbols of its parameters and of the elements bound to its roles. A predator
and prey system takes one element class, the general `RateInteraction`, and one new interaction:

```python
from typing import Annotated

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, RateInteraction
from skcomplex.core.components.laws import RateLaw
from skcomplex.core.components.systems import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec


class PopulationElement(Element):
    def __init__(self, label, visual=None, *, n: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.n = n


class PredationInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        predator: Annotated[str, RoleSpec(states=('n',))],
        prey: Annotated[str, RoleSpec(states=('n',))],
        a: Annotated[float, ParameterSpec()] = 0.4,
        b: Annotated[float, ParameterSpec()] = 0.1,
    ):
        super().__init__(label)
        self.predator = predator
        self.prey = prey
        self.a = a
        self.b = b

    def _get_laws(self, s):
        contacts = s.predator.n * s.prey.n
        return [RateLaw(s.prey.n, -s.a * contacts), RateLaw(s.predator.n, s.b * contacts)]


system = System(
    elements=[PopulationElement('rabbits', n=40), PopulationElement('foxes', n=9)],
    interactions=[
        RateInteraction('growth', target='rabbits', state='n', rate=lambda s: 1.1 * s.target.n * (1 - s.target.n / 1000)),
        PredationInteraction('hunt', predator='foxes', prey='rabbits'),
        RateInteraction('death', target='foxes', state='n', rate=lambda s: -0.4 * s.target.n),
    ],
)
trajectory = simulate(system, until=30, dt=0.05)
round(float(trajectory.values('rabbits.n').max()))
```

A `RerunObserver` records the run for the Rerun viewer, and `RerunObserver('predators', spawn=True)` opens the viewer
during the run. A schedule changes a parameter while the system runs, here halving the hunting rate at time 15:

```python continuation
import tempfile
from pathlib import Path

from skcomplex.core.engine.viz import RerunObserver

recording = Path(tempfile.mkdtemp()) / 'predators.rrd'
trajectory = simulate(
    system,
    until=30,
    dt=0.05,
    observers=[RerunObserver('predators', save_path=recording)],
    schedule={'hunt.a': lambda t: 0.4 if t < 15 else 0.2},
)
float(trajectory.times[-1])
```

Adding a process takes one more interaction, and the analysis `symbolic` shows the equations of the system:

```python continuation
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.analysis import symbolic


class HarvestInteraction(Interaction):
    def __init__(
        self,
        label,
        *,
        target: Annotated[str, RoleSpec(states=('n',))],
        h: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ):
        super().__init__(label)
        self.target = target
        self.h = h

    def _get_laws(self, s):
        return [RateLaw(s.target.n, -s.h * s.target.n)]


harvested = System(
    elements=[PopulationElement('fish', n=500)],
    interactions=[
        RateInteraction('growth', target='fish', state='n', rate=lambda s: 0.5 * s.target.n * (1 - s.target.n / 1000)),
        HarvestInteraction('fishing', target='fish', h=0.1),
    ],
)
symbolic(harvested).equations['fish.n']
```

### Find out which components fit

An interaction states what it needs from each role, and an element states what it carries. Ask them before you build:

```python continuation
hunt = PredationInteraction('hunt', predator='foxes', prey='rabbits')
hunt.get_requirements().roles['prey'].states, list(PopulationElement('rabbits').get_states())
```

When a role is bound to an element without the state it reads, `simulate` raises an error that names the role, the
state, and the fix:

```python continuation
from skcomplex.core.errors import RoleElementError


class TankElement(Element):
    def __init__(self, label, visual=None, *, level: Annotated[float, StateSpec()] = 1.0):
        super().__init__(label, visual=visual)
        self.level = level


wrong = System(
    elements=[TankElement('tank'), PopulationElement('foxes')],
    interactions=[PredationInteraction('hunt', predator='foxes', prey='tank')],
)
try:
    simulate(wrong, until=1)
except RoleElementError as error:
    message = str(error)
message
```

The [parts pages](https://georgedouzas.github.io/scikit-complexity/parts/elements/) list every element, interaction, law,
solver, and backend with the tags they accept, and the
[user guide](https://georgedouzas.github.io/scikit-complexity/overview/user_guide/build_a_custom_system/choosing_compatible_components/)
walks through the choices in order.

## Spec-driven development

The project's engineering constitution lives in `openspec/constitution.md` and binds all work. The
project uses [OpenSpec](https://github.com/Fission-AI/OpenSpec) for spec-driven development, and
`openspec/config.yaml` names the constitution as binding.
