"""Flux balance analysis of Raman's toy network, or of the Escherichia coli core system.

The Escherichia coli core system needs the extra.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.spaces import StaticTime
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.errors import ValidationError
from ..common.metabolism import ToyNetworkElement, load_network

NETWORKS = ('toy', 'ecoli')
OXYGEN_EXCHANGE = 'EX_o2_e'


class FluxBalanceSystem(BaseSystem):
    """Flux balance analysis of a metabolic network.

    It sets the fluxes of the network to maximise growth.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValidationError:
            The network is unknown.

        MissingExtraError:
            The E. coli core system needs the extra, which is not installed.

    Examples:
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.biology.systems_biology import FluxBalanceSystem
        >>> trajectory = simulate(FluxBalanceSystem(), until=0.0)
        >>> round(float(trajectory.values('cell.v_vbio')[0]), 6)
        50.0
    """

    field = 'biology/systems_biology'
    topics = ('flux balance', 'linear program', 'metabolism', 'static')
    references = ('Raman 2021', 'Orth et al. 2010')
    until = 0.0

    def __init__(
        self,
        network: str = 'toy',
        *,
        oxygen: float = 1000.0,
        parsimonious: bool = False,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.network = network
        self.oxygen = oxygen
        self.parsimonious = parsimonious

    def _build_system(self) -> CompositeSystem:
        """Return the parts of flux balance analysis of a metabolic network."""
        if self.network not in NETWORKS:
            error_msg = f'Unknown network {self.network!r}. Use one of {", ".join(NETWORKS)}.'
            raise ValidationError(error_msg)
        if self.network == 'toy':
            return CompositeSystem(
                self.label,
                elements=[ToyNetworkElement('cell', parsimonious=self.parsimonious)],
                time=StaticTime(),
            )
        element = load_network('textbook', 'cell', needed_by='flux_balance')
        element.parsimonious = self.parsimonious
        identifier = f'lower_{element._identifier(OXYGEN_EXCHANGE)}'
        element.set_params(**{identifier: -float(self.oxygen)})
        return CompositeSystem(self.label, elements=[element], time=StaticTime())
