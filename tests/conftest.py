"""Configuration for the pytest test suite."""

import importlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.base import parameter, role
from skcomplex.core.components import Element, Wiener
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import Law, Noise, Rate, Symbols, Update
from skcomplex.core.system import System


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the option that runs the timing benchmarks, which need an otherwise idle machine."""
    parser.addoption('--benchmarks', action='store_true', help='Run the timing benchmarks.')
    parser.addoption('--slow', action='store_true', help='Run the slow tests, which take more than 10 seconds.')


def pytest_configure(config: pytest.Config) -> None:
    """Register the benchmark marker, and import the package's warnings in the main thread.

    The controller of a parallel run rebuilds the warnings its workers send, and Sage imports only in a main thread.
    """
    importlib.import_module('skcomplex.core.base')
    config.addinivalue_line('markers', 'benchmark: a timing benchmark, run with --benchmarks on an idle machine')
    config.addinivalue_line('markers', 'slow: a test that takes more than 10 seconds, run with --slow')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the timing benchmarks and the slow tests unless they are requested."""
    skips = {
        'benchmark': (config.getoption('--benchmarks'), 'timing benchmark, run with --benchmarks on an idle machine'),
        'slow': (config.getoption('--slow'), 'slow test, run with --slow'),
    }
    for marker, (requested, reason) in skips.items():
        if requested:
            continue
        for item in items:
            if marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


Node = Element.define('Node', states={'x': 0.0})
Holding = Element.define('Holding', states={'n': 0.0})


def _summed(inputs: str | Sequence[str], weights: float | Sequence[float] = 1.0) -> System:
    """Return the system of a weighted sum of inputs read from three elements and a group of ten nodes."""
    elements = [
        Node('a', x=1.0),
        Node('b', x=2.0),
        Node('c', x=4.0),
        Node('hub', x=8.0),
        Node('target'),
        Node.group('nodes', 10, x=np.arange(10.0)),
    ]
    return System(
        components=elements,
        interactions=[WeightedSum('sum', target='target', inputs=inputs, weights=weights)],
    )


def _correlated(covariance: npt.ArrayLike | None) -> System:
    """Return two holdings driven by a Wiener element of a covariance."""
    return System(
        components=[Holding('a'), Holding('b'), Wiener('market', dimension=2, covariance=covariance)],
        interactions=[Drive('drive', first='a', second='b', noise='market')],
    )


class CallableGrowth(Interaction):
    """Growth whose rate is a Python callable."""

    target: str = role()
    r: float = parameter(1.0)

    def laws(self, s: Symbols) -> list[Law]:
        """Return a callable rate law."""
        name = f'{self.target}.n'
        return [Rate(s.target.n, lambda t, values: values[f'{self.label}.r'] * values[name])]


class Drive(Interaction):
    """Unit noise on two elements from the two processes of a Wiener element."""

    first: str = role()
    second: str = role()
    noise: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the noise of both elements."""
        return [Noise(s.first.n, 1.0, process=s.noise.dW[0]), Noise(s.second.n, 1.0, process=s.noise.dW[1])]


class WeightedSum(Interaction):
    """Update of a target to the weighted sum of the states of a variable number of inputs."""

    target: str = role()
    inputs: list[str] = role(variadic=True)
    weights: list[float] = parameter(1.0, per='inputs')

    def laws(self, s: Symbols) -> list[Law]:
        """Return the update of the target."""
        return [Update(s.target.x, sum((w * item.x for w, item in zip(s.weights, s.inputs, strict=True)), 0))]
