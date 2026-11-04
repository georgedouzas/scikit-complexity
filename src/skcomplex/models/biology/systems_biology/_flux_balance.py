"""Flux balance analysis of Raman's toy network, or of the E. coli core model with the extra."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.spaces import StaticTime
from ....core.system import System
from ....errors import ValidationError
from ..._catalog import register_model
from ..common.metabolism import ReactionSpec, load_network, metabolic_network_type

NETWORKS = ('toy', 'ecoli')
TOY_REACTIONS = {
    'v1': ReactionSpec({'A': -1.0, 'ATP': 1.0, 'B': 1.0}),
    'v2': ReactionSpec({'A': -1.0, 'C': 1.0}),
    'v3': ReactionSpec({'C': -1.0, 'ATP': 1.0, 'B': 1.0}),
    'v4': ReactionSpec({'B': -1.0, 'D': 2.0}, lower=-1000.0),
    'v5': ReactionSpec({'B': -1.0, 'F': 1.0}),
    'v6': ReactionSpec({'D': -1.0, 'E': 1.0}, lower=-1000.0),
    'v7': ReactionSpec({'D': -1.0, 'ATP': -1.0, 'G': 1.0}),
    'b1': ReactionSpec({'A': 1.0}),
    'b2': ReactionSpec({'E': -1.0}, lower=-1000.0),
    'vbio': ReactionSpec({'F': -1.0, 'G': -2.0, 'ATP': -10.0}),
}
"""The reactions of Table 8.2 of Raman (2021), with the default bounds of its Section 8.6."""

ToyNetworkElement = metabolic_network_type('ToyNetworkElement', TOY_REACTIONS, {'vbio': 1.0})
"""The toy network of Table 8.2 of Raman (2021), which grows on A alone."""

ToyNetworkElement.__module__ = __name__
OXYGEN_EXCHANGE = 'EX_o2_e'


@register_model(
    field='biology/systems_biology',
    tags=('flux balance', 'linear program', 'metabolism', 'static'),
    references=('Raman 2021', 'Orth et al. 2010'),
    defaults={'until': 0.0},
)
def flux_balance(
    network: str = 'toy',
    *,
    oxygen: float = 1000.0,
    parsimonious: bool = False,
) -> System:
    """Return flux balance analysis of a metabolic network, which sets its fluxes to maximise growth.

    The steady-state balances of the metabolites and the bounds of the reactions confine the fluxes, and the optimum of
    the growth flux sets them (Raman 2021, Section 8.6). The toy network of Table 8.2 reaches growth 50, set by the
    bound 1000 on `v6`. The E. coli core model needs the extra `metabolism`, and grows at 0.8739 h^-1 on glucose with
    oxygen and at 0.2117 h^-1 without.

    Args:
        network:
            `'toy'` for Raman's toy network or `'ecoli'` for the E. coli core model.

        oxygen:
            The largest oxygen uptake of the E. coli core model.

        parsimonious:
            Whether to minimise the total absolute flux after growth.

    Returns:
        system:
            The system.

    Raises:
        ValidationError:
            The network is unknown.

        MissingExtraError:
            The E. coli core model needs the extra, which is not installed.

    Examples:
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.biology.systems_biology import flux_balance
        >>> trajectory = Simulation(flux_balance()).check().prepare().run(0.0).trajectory_
        >>> round(float(trajectory.values('cell.v_vbio')[0]), 6)
        50.0
    """
    if network not in NETWORKS:
        error_msg = f'Unknown network {network!r}. Use one of {", ".join(NETWORKS)}.'
        raise ValidationError(error_msg)
    if network == 'toy':
        return System(elements=[ToyNetworkElement('cell', parsimonious=parsimonious)], time=StaticTime())
    element = load_network('textbook', 'cell', needed_by='flux_balance')
    element.parsimonious = parsimonious
    identifier = f'lower_{element._identifier(OXYGEN_EXCHANGE)}'
    setattr(element, identifier, -float(oxygen))
    return System(elements=[element], time=StaticTime())
