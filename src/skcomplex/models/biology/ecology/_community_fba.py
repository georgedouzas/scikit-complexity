"""Community flux balance analysis: two E. coli core members joined through a shared medium."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence

from ....core.spaces import StaticTime
from ....core.system import System
from ..._catalog import register_model
from ..common.metabolism import CommunityInteraction, community_medium_type, load_network

GROWTH = 'Biomass_Ecoli_core'
CROSS_FED = ('EX_akg_e', 'EX_glu__L_e')
OPEN_UPTAKE = -1000.0


@register_model(
    field='biology/ecology',
    tags=('community flux balance', 'SteadyCom', 'cross-feeding', 'linear program', 'metabolism', 'static'),
    references=('Raman 2021', 'Khandelwal et al. 2013', 'Chan et al. 2017', 'Diener et al. 2020'),
    defaults={'until': 0.0},
)
def community_fba(
    knockouts: Sequence[str] = ('ICDHyr', 'NH4t'),
    *,
    abundances: Sequence[float] = (0.5, 0.5),
    exchanges: Mapping[str, float] | None = None,
) -> System:
    """Return two members of the E. coli core model that share a medium, which needs the extra `metabolism`.

    The programs of the members merge through a medium whose law holds the growth weighted by the abundances (Raman
    2021, Section 11.1.3). By default one member lacks isocitrate dehydrogenase and takes up 2-oxoglutarate, and the
    other lacks ammonium transport and takes up glutamate, each from what the other secretes, so neither grows alone,
    and together both grow. `CommunityAnalysis` gives SteadyCom and the cooperative trade-off of the same members.

    Args:
        knockouts:
            The reaction to close in each member, or the empty string for none.

        abundances:
            The abundance of each member.

        exchanges:
            The lower bound of the net exchange of the community, by reaction, or `None` for no net uptake of the
            cross-fed metabolites.

    Returns:
        system:
            The system.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.ecology import community_fba
        >>> trajectory = Simulation(community_fba()).check().prepare().run(0.0).trajectory_
        >>> round(float(trajectory.values('medium.objective_0')[0]), 3)
        0.862
    """
    shared = dict(exchanges) if exchanges is not None else dict.fromkeys(CROSS_FED, 0.0)
    members = []
    for position, knockout in enumerate(knockouts):
        member = load_network('textbook', f'member_{position}', needed_by='community_fba')
        member.objective = {}
        for reaction in shared:
            if shared[reaction] >= 0:
                setattr(member, f'lower_{member._identifier(reaction)}', OPEN_UPTAKE)
        if knockout:
            setattr(member, f'lower_{member._identifier(knockout)}', 0.0)
            setattr(member, f'upper_{member._identifier(knockout)}', 0.0)
        members.append(member)
    return System(
        elements=[*members, community_medium_type(list(shared))('medium')],
        interactions=[
            CommunityInteraction(
                'joint',
                members=[member.label for member in members],
                medium='medium',
                abundances=tuple(abundances),
                exchanges=shared,
                growth=GROWTH,
            ),
        ],
        time=StaticTime(),
    )
