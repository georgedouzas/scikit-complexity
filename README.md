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

## Introduction

`scikit-complexity` describes models from any science as elements that carry state, interactions that change that
state through laws, and systems that nest and compose. It infers the mathematical form of a model from its laws,
simulates it offline or in real time, and streams the run to the [Rerun](https://rerun.io) viewer.

- Rate laws give ordinary differential equations, rate and noise laws give stochastic differential equations, jump
  laws give jump processes simulated with the Gillespie method, and update laws give discrete maps.
- Laws are Sage symbolic expressions or plain Python functions. Symbolic laws can be inspected and solved exactly.
- Ready-made models cover physics, ecology, epidemiology, economics, and finance.

## Installation

`scikit-complexity` supports Python 3.12 and 3.13 on Linux and macOS 13 or later, and is available on PyPI:

```bash
pip install scikit-complexity
```

The installation includes Sage through the [passagemath](https://github.com/passagemath/passagemath) distributions,
LAMMPS, Mesa, Numba, and Rerun. Windows has no passagemath wheels, so on Windows the package runs in WSL.

For development, clone the repository and install the project with [PDM](https://github.com/pdm-project/pdm):

```bash
git clone https://github.com/georgedouzas/scikit-complexity.git
cd scikit-complexity
pdm install
```

## Usage

A predator and prey model assembles from ready-made interactions:

```python
from skcomplex.core.model import System
from skcomplex.models.biology.ecology import LogisticGrowth, Mortality, Population, Predation

system = System(
    elements=[Population('rabbits', n=40), Population('foxes', n=9)],
    interactions=[
        LogisticGrowth('growth', target='rabbits', r=1.1, K=1000),
        Predation('hunt', predator='foxes', prey='rabbits', a=0.4, b=0.1),
        Mortality('death', target='foxes', d=0.4),
    ],
)
```

A simulation runs it and records the trajectory. Adding a `RerunObserver` streams the run to the Rerun viewer, and a
`RealTime` clock paces it against wall time:

```python
from skcomplex.core.runtime import RealTime, Simulation
from skcomplex.core.viz import RerunObserver

simulation = Simulation(
    system,
    dt=0.05,
    clock=RealTime(speed=20.0),
    observers=[RerunObserver('lotka-volterra', spawn=True)],
)
trajectory = simulation.run(until=30).trajectory_
print(trajectory['rabbits.n'].max())
```

A simulation started with `start` runs in the background, and its parameters can change while it runs:

```python
simulation = Simulation(system, dt=0.05, clock=RealTime(speed=20.0)).start(until=30)
simulation.update_parameters({'hunt.a': 0.2})
simulation.wait()
```

A new law takes a few lines. The namespace `s` holds the symbols of the interaction's parameters and of the elements
bound to its roles:

```python
from skcomplex.core.analysis import equations_of
from skcomplex.core.base import Domain, parameter, role
from skcomplex.core.model import Interaction, Rate


class Harvest(Interaction):
    target: str = role()
    h: float = parameter(0.1, domain=Domain.NONNEGATIVE)

    def laws(self, s):
        return [Rate(s.target.n, -s.h * s.target.n)]


harvested = System(
    elements=[Population('fish', n=500)],
    interactions=[LogisticGrowth('growth', target='fish', r=0.5, K=1000), Harvest('fishing', target='fish', h=0.1)],
)
print(equations_of(harvested)['fish.n'])
```

## Spec-driven development

The project's engineering constitution lives in `openspec/constitution.md` and binds all work. The
project uses [OpenSpec](https://github.com/Fission-AI/OpenSpec) for spec-driven development, and
`openspec/config.yaml` names the constitution as binding.
