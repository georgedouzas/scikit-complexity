"""Competition of the cliques of a winners-take-all network, gated by the reservoirs of the sites."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, atan, oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total, where
from ....core.specs import ParameterSpec, RoleSpec


class CliqueCompetitionInteraction(Interaction):
    """Competition of the cliques of a winners-take-all network, gated by the reservoirs of the sites.

    Site `i` grows by `dx_i/dt = (1 - x_i) r_i` for a positive input `r_i` and decays by `dx_i/dt = x_i r_i` for a
    negative one, with `r_i = sum_j [f_w(phi_i) w_ij + z_ij f_z(phi_j)] x_j`. The excitatory links `w_ij > 0` join the
    sites of every clique, and every other pair inhibits with `z_ij = -|z|`. The reservoir functions `f(phi) = f_min +
    (1 - f_min) (atan((phi - phi_c) / width) - atan(-phi_c / width)) / (atan((1 - phi_c) / width) - atan(-phi_c /
    width))` switch off the excitation of a site whose own reservoir is low and the inhibition by a site whose
    reservoir is low.

    Args:
        label:
            The label of the interaction.

        sites:
            The path of the group of sites.

        w:
            The excitatory links, a symmetric matrix with zeros where sites inhibit each other.

        z:
            The strength of the inhibition.

        phi_w:
            The inflection point of the reservoir function of the excitation.

        phi_z:
            The inflection point of the reservoir function of the inhibition.

        width:
            The width of the reservoir functions.

        f_w_min:
            The floor of the reservoir function of the excitation.

        f_z_min:
            The floor of the reservoir function of the inhibition.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import CliqueCompetitionInteraction
        >>> CliqueCompetitionInteraction('competition', sites='sites', w=((0.0, 0.1), (0.1, 0.0))).z
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        sites: Annotated[str, RoleSpec()],
        w: Sequence[Sequence[float]] = ((0.0,),),
        z: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        phi_w: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.7,
        phi_z: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.15,
        width: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.05,
        f_w_min: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        f_z_min: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label)
        self.sites = sites
        self.w = w
        self.z = z
        self.phi_w = phi_w
        self.phi_z = phi_z
        self.width = width
        self.f_w_min = f_w_min
        self.f_z_min = f_z_min

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growth and decay of the activities.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        site = s.sites
        excitatory = np.asarray(self.w, dtype=float)
        inhibitory = np.where(excitatory > 0, 0.0, -1.0)
        np.fill_diagonal(inhibitory, 0.0)
        excitation = self._reservoir(site.phi, s.phi_w, s.f_w_min, s.width) * total(site.x, weights=excitatory)
        inhibition = s.z * total(self._reservoir(site.phi, s.phi_z, s.f_z_min, s.width) * site.x, weights=inhibitory)
        r = excitation + inhibition
        return [RateLaw(site.x, where(r > 0, (1 - site.x) * r, site.x * r))]

    def _reservoir(self, phi: Expression, center: Expression, floor: Expression, width: Expression) -> Expression:
        """Return the sigmoidal reservoir function of a reservoir level."""
        lowest = atan(-center / width)
        return floor + (1 - floor) * (atan((phi - center) / width) - lowest) / (atan((1 - center) / width) - lowest)
