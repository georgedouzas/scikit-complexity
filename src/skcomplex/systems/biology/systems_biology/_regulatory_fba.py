"""Regulatory flux balance analysis: Boolean rules alternate with the program in a culture on glucose and acetate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ....core.engine.solvers import CoSimulationSolver, RK4Solver
from ....core.specs import RoleSpec
from ..._catalog import register_system
from ..common.metabolism import (
    BiomassElement,
    ExchangeInteraction,
    GrowthInteraction,
    SubstrateElement,
    SwitchInteraction,
    load_network,
)
from ..common.networks import BooleanNodeElement

GROWTH = 'Biomass_Ecoli_core'
GLUCOSE_EXCHANGE = 'EX_glc__D_e'
ACETATE_EXCHANGE = 'EX_ac_e'
ACETATE_GENES = ('b2296', 'b3115', 'b1849')
SYNCHRONISATION = 0.05
ODE_STEP = 0.005


class GlucoseSensorInteraction(Interaction):
    """Boolean node that reads whether a substrate lies above a threshold, or the previous node of a chain.

    Args:
        label:
            The label of the interaction.

        node:
            The path of the Boolean node.

        source:
            The path of what the node reads: a substrate with a threshold, or a Boolean node without one.

        threshold:
            The concentration above which the node turns on, or `None` for a node that copies another.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import GlucoseSensorInteraction
        >>> GlucoseSensorInteraction('sensor', node='a', source='glucose', threshold=0.1).threshold
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        node: Annotated[str, RoleSpec()] = 'node',
        source: Annotated[str, RoleSpec()] = 'source',
        threshold: float | None = None,
    ) -> None:
        super().__init__(label)
        self.node = node
        self.source = source
        self.threshold = threshold

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BaseLaw]:
        """Return the update of the node.

        Args:
            namespace:
                The namespace of the roles.

        Returns:
            laws:
                The update law of the node.
        """
        if self.threshold is None:
            return [UpdateLaw(namespace.node.x, namespace.source.x)]
        return [UpdateLaw(namespace.node.x, where(namespace.source.c > float(self.threshold), 1, 0))]


@register_system(
    field='biology/systems_biology',
    tags=('regulatory flux balance', 'boolean network', 'linear program', 'metabolism', 'diauxie', 'co-simulation'),
    references=('Raman 2021', 'Covert et al. 2001'),
    defaults={
        'until': 8.0,
        'dt': SYNCHRONISATION,
        'solver': CoSimulationSolver(sync_dt=SYNCHRONISATION, solvers={'culture': RK4Solver(dt=ODE_STEP)}),
    },
)
def regulatory_fba(
    *,
    biomass: float = 0.01,
    glucose: float = 10.0,
    acetate: float = 10.0,
    threshold: float = 0.1,
    lag: int = 10,
    acetate_vmax: float = 10.0,
    km: float = 0.015,
) -> System:
    """Return a batch culture on glucose and acetate with Boolean catabolite repression, which needs the extra.

    Boolean rules alternate with the program (Raman 2021, Eq. 9.7). A node reads whether glucose lies above a threshold,
    a chain of nodes delays it, and while the last node is on it holds the genes of acetate kinase off through update
    laws. So the culture leaves acetate alone while glucose lasts, stops at depletion, and grows again on acetate after
    the lag of the chain.

    The alternation is that of Raman, but the rules are an illustration and not those of Covert et al. (2001): the genes
    of acetate kinase, the threshold, and the chain of nodes come from no source, and the diauxie of Raman is on glucose
    and lactose, which the E. coli core system cannot take up.

    Args:
        biomass:
            The initial biomass in gDW per L.

        glucose:
            The initial glucose in mM.

        acetate:
            The initial acetate in mM.

        threshold:
            The glucose above which the sensor is on, in mM.

        lag:
            The number of nodes after the sensor, each a synchronisation interval of delay.

        acetate_vmax:
            The largest acetate uptake in mmol per gDW per h.

        km:
            The half-saturation concentration of both substrates in mM.

    Returns:
        system:
            The system.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import regulatory_fba
        >>> [system.label for system in regulatory_fba(lag=1).systems]
        ['control', 'culture']
    """
    cell = load_network('textbook', 'cell', needed_by='regulatory_fba')
    cell.parsimonious, cell.infeasible = True, 'zero'
    identifier = cell._identifier(ACETATE_EXCHANGE)
    cell.set_params(
        **{f'lower_{identifier}': -float(acetate_vmax)},
        **{cell.gene_name(gene): 0.0 for gene in ACETATE_GENES},
    )
    nodes = [BooleanNodeElement(f'node_{position}', x=1) for position in range(lag + 1)]
    chain = [
        GlucoseSensorInteraction(
            f'delay_{position}',
            node=f'control/node_{position}',
            source=f'control/node_{position - 1}',
        )
        for position in range(1, lag + 1)
    ]
    return System(
        systems=[
            System('control', elements=[cell, *nodes], time=DiscreteTime()),
            System(
                'culture',
                elements=[
                    BiomassElement('biomass', x=biomass),
                    SubstrateElement('glucose', c=glucose),
                    SubstrateElement('acetate', c=acetate),
                ],
            ),
        ],
        interactions=[
            GlucoseSensorInteraction(
                'sensor',
                node='control/node_0',
                source='culture/glucose',
                threshold=threshold,
            ),
            *chain,
            SwitchInteraction('repression', node=f'control/node_{lag}', cell='control/cell', genes=ACETATE_GENES),
            ExchangeInteraction(
                'glucose_uptake',
                cell='control/cell',
                biomass='culture/biomass',
                substrate='culture/glucose',
                reaction=GLUCOSE_EXCHANGE,
                km=km,
                interval=SYNCHRONISATION,
            ),
            ExchangeInteraction(
                'acetate_uptake',
                cell='control/cell',
                biomass='culture/biomass',
                substrate='culture/acetate',
                reaction=ACETATE_EXCHANGE,
                vmax=acetate_vmax,
                km=km,
                interval=SYNCHRONISATION,
            ),
            GrowthInteraction('growth', cell='control/cell', biomass='culture/biomass', reaction=GROWTH),
        ],
    )
