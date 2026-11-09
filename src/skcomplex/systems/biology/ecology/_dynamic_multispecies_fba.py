"""Dynamic flux balance analysis of two E. coli core strains on one medium.

The two strains compete for the medium.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.components.spaces import DiscreteTime
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import CoSimulationSolver, RK4Solver
from ..common.metabolism import (
    BiomassElement,
    ExchangeInteraction,
    GrowthInteraction,
    SubstrateElement,
    load_network,
)

GROWTH = 'Biomass_Ecoli_core'
GLUCOSE_EXCHANGE = 'EX_glc__D_e'
SYNCHRONISATION = 0.05
ODE_STEP = 0.005


class DynamicMultispeciesFBASystem(BaseSystem):
    """Two E. coli core strains that differ in maximal glucose uptake on one medium.

    It needs the extra `metabolism`.

    Each strain has its own program, uptake kinetics, and biomass, and they read one glucose pool (Raman 2021, Section
    11.1.3.4). The strain with the larger uptake grows faster, so its share rises until the glucose runs out. The two
    strains and their uptakes of 10 and 8 are an illustration of the framework, whose example in the source is the
    competition between Rhodoferax and Geobacter (Zhuang et al. 2011).

    Args:
        fast_vmax:
            The largest glucose uptake of the fast strain, in mmol per gDW per h.

        slow_vmax:
            The largest glucose uptake of the slow strain, in mmol per gDW per h.

        biomass:
            The initial biomass of each strain in gDW per L.

        glucose:
            The initial glucose in mM.

        km:
            The half-saturation concentration of glucose in mM.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.ecology import DynamicMultispeciesFBASystem
        >>> [element.label for element in DynamicMultispeciesFBASystem().prepare().system_.systems[1].elements]
        ['biomass_fast', 'biomass_slow', 'glucose']
    """

    field = 'biology/ecology'
    topics = ('dynamic flux balance', 'competition', 'linear program', 'metabolism', 'co-simulation')
    references = ('Raman 2021', 'Zhuang et al. 2011')
    until = 8.0
    dt = SYNCHRONISATION
    solver = CoSimulationSolver(sync_dt=SYNCHRONISATION, solvers={'culture': RK4Solver(dt=ODE_STEP)})

    def __init__(
        self,
        *,
        fast_vmax: float = 10.0,
        slow_vmax: float = 8.0,
        biomass: float = 0.01,
        glucose: float = 10.0,
        km: float = 0.015,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.fast_vmax = fast_vmax
        self.slow_vmax = slow_vmax
        self.biomass = biomass
        self.glucose = glucose
        self.km = km

    def _build_system(self) -> System:
        """Return the parts of two E. coli core strains that differ in glucose uptake.

        They differ in maximal glucose uptake on one medium.
        """
        cells = []
        interactions = []
        for name, vmax in (('fast', self.fast_vmax), ('slow', self.slow_vmax)):
            cell = load_network('textbook', f'cell_{name}', needed_by='dynamic_multispecies_fba')
            cell.parsimonious, cell.infeasible = True, 'zero'
            cells.append(cell)
            interactions += [
                ExchangeInteraction(
                    f'uptake_{name}',
                    cell=f'fba/cell_{name}',
                    biomass=f'culture/biomass_{name}',
                    substrate='culture/glucose',
                    reaction=GLUCOSE_EXCHANGE,
                    vmax=vmax,
                    km=self.km,
                    interval=SYNCHRONISATION,
                ),
                GrowthInteraction(
                    f'growth_{name}',
                    cell=f'fba/cell_{name}',
                    biomass=f'culture/biomass_{name}',
                    reaction=GROWTH,
                ),
            ]
        return System(
            self.label,
            systems=[
                System('fba', elements=cells, time=DiscreteTime()),
                System(
                    'culture',
                    elements=[
                        BiomassElement('biomass_fast', x=self.biomass),
                        BiomassElement('biomass_slow', x=self.biomass),
                        SubstrateElement('glucose', c=self.glucose),
                    ],
                ),
            ],
            interactions=interactions,
        )
