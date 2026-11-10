"""Community flux balance analysis of two E. coli core members.

The members are joined through a shared medium.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ....core.components.spaces import StaticTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ..common.metabolism import CommunityInteraction, community_medium_type, load_network

GROWTH = 'Biomass_Ecoli_core'
CROSS_FED = ('EX_akg_e', 'EX_glu__L_e')
OPEN_UPTAKE = -1000.0


class CommunityFBASystem(BaseSystem):
    """Two members of the E. coli core system that share a medium.

    It needs the extra `metabolism`.

    The programs of the members merge through a medium whose law holds the growth weighted by the abundances (Raman
    2021, Section 11.1.3). By default one member lacks isocitrate dehydrogenase and takes up 2-oxoglutarate, and the
    other lacks ammonium transport and takes up glutamate, each from what the other secretes, so neither grows alone,
    and together both grow. `community_growth` gives SteadyCom and the cooperative trade-off of the same members. The
    pair is an illustration found on the E. coli core system: the consortium of Raman (Section 11.1.3.6) has four
    mutants that need amino acids, whose synthesis the core system does not hold.

    Args:
        knockouts:
            The reaction to close in each member, or the empty string for none.

        abundances:
            The abundance of each member.

        exchanges:
            The lower bound of the net exchange of the community, by reaction, or `None` for no net uptake of the
            cross-fed metabolites.

        common_rate:
            Whether the members grow at one specific rate, as in community FBA, rather than each at its own, as in joint
            FBA.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.biology.ecology import CommunityFBASystem
        >>> trajectory = simulate(CommunityFBASystem(), until=0.0)
        >>> round(float(trajectory.get_values('medium.objective_0')[0]), 3)
        0.862
    """

    field = 'biology/ecology'
    topics = ('community flux balance', 'SteadyCom', 'cross-feeding', 'linear program', 'metabolism', 'static')
    references = ('Raman 2021', 'Khandelwal et al. 2013', 'Chan et al. 2017', 'Diener et al. 2020')
    until = 0.0

    def __init__(
        self,
        knockouts: Sequence[str] = ('ICDHyr', 'NH4t'),
        *,
        abundances: Sequence[float] = (0.5, 0.5),
        exchanges: Mapping[str, float] | None = None,
        common_rate: bool = False,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.knockouts = knockouts
        self.abundances = abundances
        self.exchanges = exchanges
        self.common_rate = common_rate

    def _build_system(self) -> CompositeSystem:
        """Return the parts of two members of the E. coli core system that share a medium."""
        shared = dict(self.exchanges) if self.exchanges is not None else dict.fromkeys(CROSS_FED, 0.0)
        members = []
        for position, knockout in enumerate(self.knockouts):
            member = load_network('textbook', f'member_{position}', needed_by='community_fba')
            member.objective = {}
            for reaction in shared:
                if shared[reaction] >= 0:
                    member.set_params(**{f'lower_{member._identifier(reaction)}': OPEN_UPTAKE})
            if knockout:
                identifier = member._identifier(knockout)
                member.set_params(**{f'lower_{identifier}': 0.0, f'upper_{identifier}': 0.0})
            members.append(member)
        return CompositeSystem(
            self.label,
            elements=[*members, community_medium_type(list(shared))('medium')],
            interactions=[
                CommunityInteraction(
                    'joint',
                    members=[member.label for member in members],
                    medium='medium',
                    abundances=tuple(self.abundances),
                    exchanges=shared,
                    growth=GROWTH,
                    common_rate=self.common_rate,
                ),
            ],
            time=StaticTime(),
        )
