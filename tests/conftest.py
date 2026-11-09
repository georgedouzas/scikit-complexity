"""Configuration for the pytest test suite."""

import importlib
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pytest
import qutip
from sage.all__sagemath_symbolics import CC

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import BaseLaw, NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import mean, total
from skcomplex.core.components.systems import System
from skcomplex.core.specs import HilbertSpec, ParameterSpec, RoleSpec, StateSpec, VisualSpec, WienerProcessSpec


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the option that runs the timing benchmarks, which need an otherwise idle machine."""
    parser.addoption('--benchmarks', action='store_true', help='Run the timing benchmarks.')
    parser.addoption('--slow', action='store_true', help='Run the slow tests, which take more than 10 seconds.')
    parser.addoption('--network', action='store_true', help='Run the tests that download from the network.')


def pytest_configure(config: pytest.Config) -> None:
    """Register the benchmark marker, and import the package's warnings in the main thread.

    The controller of a parallel run rebuilds the warnings its workers send, and Sage imports only in a main thread.
    """
    importlib.import_module('skcomplex.core.components.base')
    config.addinivalue_line('markers', 'benchmark: a timing benchmark, run with --benchmarks on an idle machine')
    config.addinivalue_line('markers', 'slow: a test that takes more than 10 seconds, run with --slow')
    config.addinivalue_line('markers', 'network: a test that downloads from the network, run with --network')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the timing benchmarks and the slow tests unless they are requested."""
    skips = {
        'benchmark': (config.getoption('--benchmarks'), 'timing benchmark, run with --benchmarks on an idle machine'),
        'slow': (config.getoption('--slow'), 'slow test, run with --slow'),
        'network': (config.getoption('--network'), 'downloads from the network, run with --network'),
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


class WeightedInputInteraction(Interaction):
    """Rate of every member of a target group, the sum or mean of a source group weighted by a matrix parameter."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        W: Annotated[object, ParameterSpec(shape=('target', 'source'))] = 0.0,
        reduce: str = 'sum',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.W = W
        self.reduce = reduce

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the weighted input of the target."""
        reduction = total if self.reduce == 'sum' else mean
        return [RateLaw(s.target.x, reduction(s.source.x, weights=s.W))]


class AdaptiveInputInteraction(WeightedInputInteraction):
    """Weighted input through an adaptive matrix parameter."""

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        source: Annotated[str, RoleSpec()],
        W: Annotated[object, ParameterSpec(shape=('target', 'source'), adaptive=True)] = 0.0,
        reduce: str = 'sum',
    ) -> None:
        super().__init__(label, target=target, source=source, W=W, reduce=reduce)


def _weighted(
    weights: npt.ArrayLike,
    inputs: npt.ArrayLike,
    *,
    source: str = 'inputs',
    reduce: str = 'sum',
    interaction: type[WeightedInputInteraction] = WeightedInputInteraction,
) -> System:
    """Return a hidden group whose rates read a source group through a matrix parameter."""
    matrix, values = np.asarray(weights, dtype=float), np.asarray(inputs, dtype=float)
    return System(
        'net',
        elements=[
            GroupElement('hidden', element=NodeElement, size=matrix.shape[0]),
            GroupElement('inputs', element=NodeElement, size=len(values), values={'x': values}),
            NodeElement('single'),
        ],
        interactions=[interaction('input', target='hidden', source=source, W=matrix, reduce=reduce)],
    )


def _spin_space(j: float = 0.5) -> HilbertSpec:
    """Return the Hilbert space of a spin `j`, with the spin components and the ladder operators, from up to down."""
    size = int(2 * j + 1)
    operators = {
        'x': (qutip.jmat(j, 'x').full(), True),
        'y': (qutip.jmat(j, 'y').full(), True),
        'z': (qutip.jmat(j, 'z').full(), True),
        'plus': (qutip.jmat(j, '+').full(), False),
        'minus': (qutip.jmat(j, '-').full(), False),
    }
    return HilbertSpec(size, operators, basis={'up': 0, 'down': size - 1})


SPIN_HALF = _spin_space()


class SpinElement(Element):
    """Spin of the Hilbert space it takes, by default a spin one half."""

    def __init__(
        self,
        label: str,
        psi: Annotated[object, StateSpec(domain=CC)] = 'up',
        *,
        hilbert: HilbertSpec = SPIN_HALF,
        visual: VisualSpec | None = VisualSpec('expectations'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.psi = psi
        self.hilbert = hilbert
