"""Nicholson-Bailey parasitism of hosts by parasitoids in discrete generations."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import exp, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.specs import ParameterSpec, RoleSpec


class ParasitismInteraction(Interaction):
    """Nicholson-Bailey parasitism of hosts by parasitoids in discrete generations.

    Parasitoids search at random, so a host escapes with probability `exp(-a P)`, giving `H' = k H exp(-a P)` and
    `P' = c H (1 - exp(-a P))`.

    Args:
        label:
            The label of the interaction.

        host:
            The path of the host population `H`.

        parasitoid:
            The path of the parasitoid population `P`.

        k:
            The reproductive rate of the hosts.

        a:
            The searching efficiency of the parasitoids.

        c:
            The number of viable eggs a parasitoid lays in a host.

    Examples:
        >>> from skcomplex.systems.biology.ecology import ParasitismInteraction
        >>> ParasitismInteraction('parasitism', host='moths', parasitoid='wasps', k=2.0).k
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        host: Annotated[str, RoleSpec()],
        parasitoid: Annotated[str, RoleSpec()],
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label)
        self.host = host
        self.parasitoid = parasitoid
        self.k = k
        self.a = a
        self.c = c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next generations `k H exp(-a P)` and `c H (1 - exp(-a P))`.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        hosts, parasitoids = s.host.n, s.parasitoid.n
        escape = exp(-s.a * parasitoids)
        return [UpdateLaw(hosts, s.k * hosts * escape), UpdateLaw(parasitoids, s.c * hosts * (1 - escape))]
