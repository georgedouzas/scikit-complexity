"""Logistic map of an aggregate, `x_t(t + 1) = r a (1 - a)` with `a = (1 - kappa) x_t + kappa x_s`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import MapInteraction
from ....core.specs import ParameterSpec, RoleSpec


class AggregateAveragingInteraction(MapInteraction):
    """Logistic map of an aggregate, `x_t(t + 1) = r a (1 - a)` with `a = (1 - kappa) x_t + kappa x_s`.

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
    ) -> None:
        super().__init__(label, target=target, state='x', function=self._average)
        self.source = source
        self.r = r
        self.kappa = kappa

    def _average(self, namespace: SimpleNamespace) -> Expression:
        """Return the logistic map `r a (1 - a)` of the aggregate `a = (1 - kappa) x_t + kappa x_s`."""
        aggregate = (1 - namespace.kappa) * namespace.target.x + namespace.kappa * namespace.source.x
        return namespace.r * aggregate * (1 - aggregate)
