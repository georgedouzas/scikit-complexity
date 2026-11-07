"""Flux balance analysis of Raman's toy network, or of the Escherichia coli core system with the extra."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.spaces import StaticTime
from ....core.components.system import System
from ....core.errors import ValidationError
from ..._catalog import register_system
from ..common.metabolism import ToyNetworkElement, load_network

NETWORKS = ('toy', 'ecoli')
OXYGEN_EXCHANGE = 'EX_o2_e'


@register_system(
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
    bound 1000 on `v6`. The E. coli core system needs the extra `metabolism`, and grows at 0.8739 h^-1 on glucose with
    oxygen and at 0.2117 h^-1 without.

    Args:
        network:
            `'toy'` for Raman's toy network or `'ecoli'` for the E. coli core system.

        oxygen:
            The largest oxygen uptake of the E. coli core system.

        parsimonious:
            Whether to minimise the total absolute flux after growth.

    Returns:
        system:
            The system.

    Raises:
        ValidationError:
            The network is unknown.

        MissingExtraError:
            The E. coli core system needs the extra, which is not installed.

    Examples:
        >>> from skcomplex.core.engine.runtime import Simulation
        >>> from skcomplex.systems.biology.systems_biology import flux_balance
        >>> trajectory = Simulation(flux_balance()).prepare().run(0.0).trajectory_
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
    element.set_params(**{identifier: -float(oxygen)})
    return System(elements=[element], time=StaticTime())
