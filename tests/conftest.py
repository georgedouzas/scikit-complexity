"""Configuration for the pytest test suite."""

import importlib
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction, WienerProcessSpec
from skcomplex.core.components.laws import NoiseLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import mean, total
from skcomplex.core.components.spaces import BaseSpace
from skcomplex.core.engine.runtime import Trajectory
from skcomplex.core.specs import VisualSpec
from skcomplex.core.systems import System
from skcomplex.core.types import Role

QUEUE_LENGTH = 2


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


class BodyElement(Element):
    """Body at the vector position `x` with the mass `m`, whose velocity is the value `x'`."""

    visual = VisualSpec('points', {'position': 'x'})

    def __init__(
        self,
        label: str,
        x: npt.ArrayLike = 0.0,
        *,
        m: Annotated[float, RealSet.open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x
        self.m = m


class NodeElement(Element):
    """Node element with the value `x`."""

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.x = x


class HoldingElement(Element):
    """Holding element with the amount `n`."""

    def __init__(
        self,
        label: str,
        *,
        n: object = 0.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.n = n


def _summed(inputs: str | Sequence[str], weights: float | Sequence[float] = 1.0) -> System:
    """Return the system of a weighted sum of inputs read from three elements and a group of ten nodes."""
    elements = [
        NodeElement('a', x=1.0),
        NodeElement('b', x=2.0),
        NodeElement('c', x=4.0),
        NodeElement('hub', x=8.0),
        NodeElement('target'),
        NodeElement('nodes', x=np.arange(10.0), size=10),
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

    def __init__(self, label: str, *, target: Role, r: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set a callable rate law."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            name = f'{self.target}.n'
            self.laws_ = [RateLaw(context.target.n, lambda t, values: values[f'{self.label}.r'] * values[name])]


class DriveInteraction(Interaction):
    """Unit noise on two elements from the two processes of a `WienerProcessSpec`."""

    def __init__(self, label: str, *, first: Role, second: Role, noise: WienerProcessSpec) -> None:
        super().__init__(label)
        self.first = first
        self.second = second
        self.noise = noise

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the noise of both elements."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            self.laws_ = [
                NoiseLaw(s.first.n, 1.0, process=s.noise.dW[0]),
                NoiseLaw(s.second.n, 1.0, process=s.noise.dW[1]),
            ]


class WeightedSumInteraction(Interaction):
    """Update of a target to the weighted sum of the states of a variable number of inputs."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        inputs: tuple[Role, ...],
        weights: float | Sequence[float] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.inputs = inputs
        self.weights = weights

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update of the target."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            weights = s.weights if isinstance(self.weights, Sequence) else [s.weights] * len(s.inputs)
            summed = sum((w * item.x for w, item in zip(weights, s.inputs, strict=True)), 0)
            self.laws_ = [UpdateLaw(s.target.x, summed)]


class WeightedInputInteraction(Interaction):
    """Rate of every member of a target group, the sum or mean of a source group weighted by a matrix parameter."""

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        source: Role,
        W: npt.ArrayLike = 0.0,
        reduce: str = 'sum',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.W = W
        self.reduce = reduce

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the weighted input of the target."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            reduction = total if self.reduce == 'sum' else mean
            self.laws_ = [RateLaw(context.target.x, reduction(context.source.x, weights=context.W))]


def _weighted(
    weights: npt.ArrayLike,
    inputs: npt.ArrayLike,
    *,
    source: str = 'inputs',
    reduce: str = 'sum',
) -> System:
    """Return a hidden group whose rates read a source group through a matrix parameter."""
    matrix, values = np.asarray(weights, dtype=float), np.asarray(inputs, dtype=float)
    return System(
        'net',
        elements=[
            NodeElement('hidden', size=matrix.shape[0]),
            NodeElement('inputs', x=values, size=len(values)),
            NodeElement('single'),
        ],
        interactions=[WeightedInputInteraction('input', target='hidden', source=source, W=matrix, reduce=reduce)],
    )


class GrowthInteraction(Interaction):
    """Growth of the amount `n` of a target at the rate `r`."""

    def __init__(self, label: str, *, target: Role, r: float = 0.0) -> None:
        super().__init__(label)
        self.target = target
        self.r = r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the growth of the amount."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.target.n, context.r * context.target.n)]


def _total(s: SimpleNamespace) -> object:
    """Return the sum of the two populations."""
    return s.a.n + s.b.n


def _two_populations() -> System:
    """Return two populations, the first growing at rate 1."""
    return System(
        elements=[HoldingElement('a', n=2.0), HoldingElement('b', n=6.0)],
        interactions=[GrowthInteraction('growth_a', target='a', r=1.0), GrowthInteraction('growth_b', target='b')],
    )


class QueueInteraction(Interaction):
    """Rule of agents with a queue of pending orders that they append to and pop from, each order an id."""

    reports: ClassVar[tuple[str, ...]] = ('longest',)

    def __init__(self, label: str, *, target: Role) -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rule of the queue."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [UpdateLaw(context.target, self._step)]

    def _step(self, agent: SimpleNamespace, system: SimpleNamespace) -> None:
        """Append the order of the step, and pop the oldest of a full queue."""
        orders = (*agent.orders, int(system.time))
        agent.orders = orders[1:] if len(orders) > QUEUE_LENGTH else orders
        system.report('longest', tuple(sorted(orders)))


def _queue_run(until: int = 3) -> Trajectory:
    """Return the trajectory of two queues."""
    orders = np.empty(2, dtype=object)
    orders[:] = [(), (7,)]
    queues = Element('queues', values={'orders': orders}, size=2)
    system = System(elements=[queues], interactions=[QueueInteraction('queue', target='queues')])
    return system.prepare().simulate(until=until, random_state=0)
