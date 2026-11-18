"""Regulatory flux balance analysis of a culture on glucose and acetate.

Boolean rules alternate with the program.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ....core.components import Observable
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import where
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, CoSimulationSolver, RK4Solver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.metabolism import (
    BiomassElement,
    ExchangeInteraction,
    GrowthInteraction,
    MetabolicCellElement,
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
    """Boolean node that reads whether a substrate lies above a threshold.

    It reads the previous node instead when it belongs to a chain.

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
        node: Role = 'node',
        source: Role = 'source',
        threshold: float | None = None,
    ) -> None:
        super().__init__(label)
        self.node = node
        self.source = source
        self.threshold = threshold

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the update of the node.

        Args:
            context:
                The namespace of the node and its source.
        """
        super()._prepare(context)
        if not isinstance(context, SimpleNamespace):
            return
        if self.threshold is None:
            self.laws_: list[BaseLaw] = [UpdateLaw(context.node.x, context.source.x)]
        else:
            self.laws_ = [UpdateLaw(context.node.x, where(context.source.c > float(self.threshold), 1, 0))]


class RegulatoryFBASystem(BaseSystem):
    """A batch culture on glucose and acetate with Boolean catabolite repression.

    It needs the extra.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import RegulatoryFBASystem
        >>> [system.label for system in RegulatoryFBASystem(lag=1).prepare().system_.systems]
        ['control', 'culture']
    """

    field = 'biology/systems_biology'
    topics = ('regulatory flux balance', 'boolean network', 'linear program', 'metabolism', 'diauxie', 'co-simulation')
    references = ('Raman 2021', 'Covert et al. 2001')
    until = 8.0
    dt = SYNCHRONISATION

    def __init__(
        self,
        *,
        biomass: float = 0.01,
        glucose: float = 10.0,
        acetate: float = 10.0,
        threshold: float = 0.1,
        lag: int = 10,
        acetate_vmax: float = 10.0,
        km: float = 0.015,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.biomass = biomass
        self.glucose = glucose
        self.acetate = acetate
        self.threshold = threshold
        self.lag = lag
        self.acetate_vmax = acetate_vmax
        self.km = km

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of a batch culture with Boolean catabolite repression.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        network = load_network('textbook', 'cell', needed_by='regulatory_fba').set_params(
            parsimonious=True,
            infeasible='zero',
        )
        identifier = network._get_identifier(ACETATE_EXCHANGE)
        values = network.get_values() | {
            f'lower_{identifier}': -float(self.acetate_vmax),
            **{network.get_gene_name(gene): 0.0 for gene in ACETATE_GENES},
        }
        cell = MetabolicCellElement('cell', values=values)
        nodes = [BooleanNodeElement(f'node_{position}', x=1) for position in range(self.lag + 1)]
        chain = [
            GlucoseSensorInteraction(
                f'delay_{position}',
                node=f'control/node_{position}',
                source=f'control/node_{position - 1}',
            )
            for position in range(1, self.lag + 1)
        ]
        self.system_ = System(
            self.label,
            systems=[
                System('control', elements=[cell, *nodes], interactions=[network]),
                System(
                    'culture',
                    elements=[
                        BiomassElement('biomass', x=self.biomass),
                        SubstrateElement('glucose', c=self.glucose),
                        SubstrateElement('acetate', c=self.acetate),
                    ],
                ),
            ],
            interactions=[
                GlucoseSensorInteraction(
                    'sensor',
                    node='control/node_0',
                    source='culture/glucose',
                    threshold=self.threshold,
                ),
                *chain,
                SwitchInteraction(
                    'repression',
                    node=f'control/node_{self.lag}',
                    cell='control/cell',
                    genes=ACETATE_GENES,
                ),
                ExchangeInteraction(
                    'glucose_uptake',
                    cell='control/cell',
                    biomass='culture/biomass',
                    substrate='culture/glucose',
                    reaction=GLUCOSE_EXCHANGE,
                    km=self.km,
                    interval=SYNCHRONISATION,
                ),
                ExchangeInteraction(
                    'acetate_uptake',
                    cell='control/cell',
                    biomass='culture/biomass',
                    substrate='culture/acetate',
                    reaction=ACETATE_EXCHANGE,
                    vmax=self.acetate_vmax,
                    km=self.km,
                    interval=SYNCHRONISATION,
                ),
                GrowthInteraction('growth', cell='control/cell', biomass='culture/biomass', reaction=GROWTH),
            ],
            solver=CoSimulationSolver(sync_dt=SYNCHRONISATION, solvers={'culture': RK4Solver(dt=ODE_STEP)}),
        )
