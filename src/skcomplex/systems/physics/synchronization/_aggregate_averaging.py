"""Logistic or Bernoulli map of an aggregate of the value of a target and the value of a source, possibly delayed."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import MapInteraction
from ....core.components.laws.utils import lag, mod
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec

MAPS = ('logistic', 'bernoulli')


class AggregateAveragingInteraction(MapInteraction):
    """Logistic or Bernoulli map of an aggregate of the value of a target and the value of a source, possibly delayed.

    The map sets `x_t(t + 1) = f(a)` with the aggregate `a = (1 - kappa) x_t(t) + kappa x_s(t - delay)`, and with the
    logistic map `f(a) = r a (1 - a)` or the Bernoulli shift `f(a) = r a mod 1`. A delayed source keeps `delay` lags.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        source:
            The path of the unit whose value the aggregate reads.

        target:
            The path of the unit whose value the map sets.

        r:
            The growth rate of the logistic map.

        kappa:
            The weight of the source in the aggregate.

        kind:
            The map, `'logistic'` or `'bernoulli'`.

        delay:
            The number of steps by which the source is delayed.

    Raises:
        ValidationError:
            The map is unknown, or the delay is not a nonnegative integer.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.synchronization import AggregateAveragingInteraction
        >>> averaging = AggregateAveragingInteraction('averaging', source='a', target='b').prepare()
        >>> source, target = SimpleNamespace(x=var('u')), SimpleNamespace(x=var('v'))
        >>> namespace = SimpleNamespace(source=source, target=target, r=4, kappa=var('kappa'))
        >>> [law.value.factor() for law in averaging.run(namespace).laws_]
        [-4*(kappa*u - kappa*v + v)*(kappa*u - kappa*v + v - 1)]
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        kind: str = 'logistic',
        delay: int = 0,
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._average)
        self.source = source
        self.r = r
        self.kappa = kappa
        self.kind = kind
        self.delay = delay

    def _check(self) -> None:
        """Raise an error for an unknown map or a delay that is not a nonnegative integer.

        Raises:
            ValidationError:
                The map is unknown, or the delay is not a nonnegative integer.
        """
        super()._check()
        if self.kind not in MAPS:
            error_msg = f'Unknown map {self.kind!r}. Use one of {", ".join(MAPS)}.'
            raise ValidationError(error_msg)
        if isinstance(self.delay, bool) or not isinstance(self.delay, int) or self.delay < 0:
            error_msg = f'The delay of {self.label!r} is {self.delay!r}. Give a nonnegative integer.'
            raise ValidationError(error_msg)

    def _average(self, namespace: SimpleNamespace) -> Expression:
        """Return the map of the aggregate `a = (1 - kappa) x_t + kappa x_s(t - delay)`."""
        source = lag(namespace.source.x, self.delay)
        aggregate = (1 - namespace.kappa) * namespace.target.x + namespace.kappa * source
        if self.kind == 'bernoulli':
            return mod(namespace.r * aggregate, 1)
        return namespace.r * aggregate * (1 - aggregate)
