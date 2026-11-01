"""Configuration for the pytest test suite."""

import importlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest

from skcomplex.core.elements import Element, GroupElement
from skcomplex.core.interactions import Interaction
from skcomplex.core.laws import BaseLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.system import System
from skcomplex.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec, WienerProcessSpec


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


class NodeElement(Element):
    """Node element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


class HoldingElement(Element):
    """Holding element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n


def _summed(inputs: str | Sequence[str], weights: float | Sequence[float] = 1.0) -> System:
    """Return the system of a weighted sum of inputs read from three elements and a group of ten nodes."""
    elements = [
        NodeElement('a', x=1.0),
        NodeElement('b', x=2.0),
        NodeElement('c', x=4.0),
        NodeElement('hub', x=8.0),
        NodeElement('target'),
        GroupElement('nodes', element=NodeElement, size=10, values={'x': np.arange(10.0)}),
    ]
    return System(
        elements=elements,
        interactions=[WeightedSumInteraction('sum', target='target', inputs=inputs, weights=weights)],
    )


def _correlated(covariance: npt.ArrayLike | None) -> System:
    """Return two holdings driven by a `WienerProcessSpec` of a covariance."""
    return System(
        elements=[HoldingElement('a'), HoldingElement('b')],
        interactions=[
            DriveInteraction(
                'drive',
                first='a',
                second='b',
                noise=WienerProcessSpec('market', dimension=2, covariance=covariance),
            ),
        ],
    )


class CallableGrowthInteraction(Interaction):
    """Growth whose rate is a Python callable."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return a callable rate law."""
        name = f'{self.target}.n'
        return [RateLaw(s.target.n, lambda t, values: values[f'{self.label}.r'] * values[name])]


class DriveInteraction(Interaction):
    """Unit noise on two elements from the two processes of a `WienerProcessSpec`."""

    def __init__(
        self,
        label: str,
        *,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        noise: Annotated[WienerProcessSpec, RoleSpec(noise=True)],
    ) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.noise = noise

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the noise of both elements."""
        return [NoiseLaw(s.first.n, 1.0, process=s.noise.dW[0]), NoiseLaw(s.second.n, 1.0, process=s.noise.dW[1])]


class WeightedSumInteraction(Interaction):
    """Update of a target to the weighted sum of the states of a variable number of inputs."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        inputs: Annotated[str | Sequence[str], RoleSpec(variadic=True)],
        weights: Annotated[float | Sequence[float], ParameterSpec(per='inputs')] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.weights = weights

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the target."""
        return [UpdateLaw(s.target.x, sum((w * item.x for w, item in zip(s.weights, s.inputs, strict=True)), 0))]
