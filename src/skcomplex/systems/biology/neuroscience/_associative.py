"""Neuroscience systems: associative thought processes.

Thoughts are transient attractors of clique-encoding networks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from itertools import combinations
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, atan, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.components.laws.utils import total, where
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

UNIT = RealSet.closed(0, 1)
CLIQUES = ((0, 1), (0, 6), (3, 6), (1, 2, 3), (4, 5, 6), (1, 2, 4, 5))


class AssociativeSiteElement(Element):
    """Site of a dense homogeneous associative network.

    The site holds an activity coupled to a slow reservoir.

    The reservoir `phi` of an active site, `x > x_c`, depletes as `dphi/dt = -gamma_minus phi`, and the reservoir of an
    inactive site recovers as `dphi/dt = gamma_plus (1 - phi) (1 - x / x_c)`. The network couples the activities.

    Args:
        label:
            The label of the site or of the group of sites.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial activity.

        phi:
            The initial reservoir.

        x_c:
            The activity above which a site counts as active.

        gamma_plus:
            The recovery rate of the reservoir.

        gamma_minus:
            The depletion rate of the reservoir.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AssociativeSiteElement
        >>> AssociativeSiteElement('sites').x_c
        0.85
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=UNIT)] = 0.0,
        phi: Annotated[float, StateSpec(domain=UNIT)] = 1.0,
        x_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.85,
        gamma_plus: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
        gamma_minus: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.phi = phi
        self.x_c = x_c
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the depletion and the recovery of the reservoir.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        recovery = s.gamma_plus * (1 - s.phi) * (1 - s.x / s.x_c)
        return [RateLaw(s.phi, where(s.x > s.x_c, -s.gamma_minus * s.phi, recovery))]


class CliqueCompetitionInteraction(Interaction):
    """Competition of the cliques of a winners-take-all network.

    The reservoirs of the sites gate the competition.

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

    law_classes = (RateLaw,)

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


class AssociativeThoughtProcessSystem(BaseSystem):
    """The dense homogeneous associative network of Gros.

    Its memories become active one after the other.

    Every memory is a clique of sites joined by excitatory links of strength `w`, and every other pair of sites inhibits
    each other with strength `z`. Without reservoirs the network is a winners-take-all network whose cliques are stable
    attractors. The reservoirs of an active clique deplete slowly, and once they are low the clique loses its excitation
    and its hold on the other sites, so the activity hops to an associated clique that shares sites with it. The
    memories become transient attractors, and the network runs through a sequence of them, an associative thought
    process. The default is the network of seven sites and six cliques of Gros (2015, figure 10.12). The system has the
    activities `sites.x` and the reservoirs `sites.phi`.

    Args:
        cliques:
            The memories, each a sequence of site indices.

        w:
            The strength of the excitatory links within a clique.

        z:
            The strength of the inhibition between sites of no common clique.

        x_c:
            The activity above which a site counts as active.

        gamma_plus:
            The recovery rate of the reservoirs.

        gamma_minus:
            The depletion rate of the reservoirs.

        start:
            The sites that are active at the start.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import AssociativeThoughtProcessSystem
        >>> AssociativeThoughtProcessSystem().prepare().system_.elements[0].prepare().size_
        7
    """

    field = 'biology/neuroscience'
    topics = ('associative memory', 'transient attractors', 'winners-take-all', 'cognitive systems', 'ode')
    references = ('Gros 2007', 'Gros 2015')
    until = 1500.0
    dt = 2.0

    def __init__(
        self,
        cliques: Sequence[Sequence[int]] = CLIQUES,
        *,
        w: float = 0.15,
        z: float = 1.0,
        x_c: float = 0.85,
        gamma_plus: float = 0.005,
        gamma_minus: float = 0.005,
        start: Sequence[int] = (0, 1),
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.cliques = cliques
        self.w = w
        self.z = z
        self.x_c = x_c
        self.gamma_plus = gamma_plus
        self.gamma_minus = gamma_minus
        self.start = start

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the dense homogeneous associative network of Gros."""
        size = 1 + max(max(clique) for clique in self.cliques)
        links = np.zeros((size, size))
        for clique in self.cliques:
            for first, second in combinations(clique, 2):
                links[first, second] = links[second, first] = self.w
        activities = np.zeros(size)
        activities[list(self.start)] = 1.0
        sites = GroupElement(
            'sites',
            element=AssociativeSiteElement,
            size=size,
            values={
                'x': activities,
                'phi': 1.0,
                'x_c': self.x_c,
                'gamma_plus': self.gamma_plus,
                'gamma_minus': self.gamma_minus,
            },
        )
        competition = CliqueCompetitionInteraction(
            'competition',
            sites='sites',
            w=tuple(map(tuple, links.tolist())),
            z=self.z,
        )
        return CompositeSystem(self.label, elements=[sites], interactions=[competition])
