"""Dephasing of a particle in the position basis, one dissipator per site projector at the rate `gamma`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, DissipatorLaw
from ....core.laws.utils import local_operator
from ....specs import ParameterSpec, RoleSpec


class PositionDephasingInteraction(Interaction):
    """Dephasing of a particle in the position basis, one dissipator per site projector at the rate `gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle, whose Hilbert space holds the site projectors `site_<k>`.

        gamma:
            The dephasing rate.

        n_sites:
            The number of sites.

    Examples:
        >>> from skcomplex.models.physics.waves import PositionDephasingInteraction
        >>> PositionDephasingInteraction('dephasing', target='packet', gamma=0.5).gamma
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        n_sites: int = 1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.n_sites = n_sites

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the dissipators of the site projectors.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The dissipators.
        """
        return [DissipatorLaw(local_operator(s.target.psi, f'site_{site}'), s.gamma) for site in range(self.n_sites)]
