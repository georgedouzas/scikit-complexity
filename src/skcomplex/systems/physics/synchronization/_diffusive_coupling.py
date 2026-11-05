"""Diffusive coupling of a state of one element to the same state of another, `x_t' += k (x_s - x_t)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import RateInteraction
from ....core.specs import ParameterSpec, RoleSpec


class DiffusiveCouplingInteraction(RateInteraction):
    """Diffusive coupling of a state of one element to the same state of another, `x_t' += k (x_s - x_t)`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        source:
            The path of the element whose state the coupling reads.

        target:
            The path of the element whose state the coupling changes.

        state:
            The name of the coupled state.

        k:
            The coupling strength.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.synchronization import DiffusiveCouplingInteraction
        >>> coupling = DiffusiveCouplingInteraction('coupling', source='a', target='b').prepare()
        >>> source, target = SimpleNamespace(x2=var('u')), SimpleNamespace(x2=var('v'))
        >>> [law.value for law in coupling.run(SimpleNamespace(source=source, target=target, k=var('k'))).laws_]
        [k*(u - v)]
    """

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        target: Annotated[str, RoleSpec()],
        state: str = 'x2',
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.source = source
        self.k = k

    def _get_rate(self, namespace: SimpleNamespace) -> Expression:
        """Return `k (x_s - x_t)`.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The rate of the state of the target.
        """
        return namespace.k * (getattr(namespace.source, self.state) - getattr(namespace.target, self.state))
