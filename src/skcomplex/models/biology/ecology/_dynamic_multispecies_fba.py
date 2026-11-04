"""Dynamic flux balance analysis of two strains of the E. coli core model that compete for one medium."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.spaces import DiscreteTime
from ....core.system import System
from ....engine.solvers import CoSimulationSolver, RK4Solver
from ..._catalog import register_model
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


@register_model(
    field='biology/ecology',
    tags=('dynamic flux balance', 'competition', 'linear program', 'metabolism', 'co-simulation'),
    references=('Raman 2021', 'Zhuang et al. 2011'),
    defaults={
        'until': 8.0,
        'dt': SYNCHRONISATION,
        'solver': CoSimulationSolver(sync_dt=SYNCHRONISATION, solvers={'culture': RK4Solver(dt=ODE_STEP)}),
    },
)
def dynamic_multispecies_fba(
    *,
    fast_vmax: float = 10.0,
    slow_vmax: float = 8.0,
    biomass: float = 0.01,
    glucose: float = 10.0,
    km: float = 0.015,
) -> System:
    """Return two E. coli core strains that differ in maximal glucose uptake on one medium, which needs the extra.

    Each strain has its own program, uptake kinetics, and biomass, and they read one glucose pool (Raman 2021, Section
    11.1.3.4). The strain with the larger uptake grows faster, so its share rises until the glucose runs out.

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

    Returns:
        system:
            The system.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.models.biology.ecology import dynamic_multispecies_fba
        >>> [element.label for element in dynamic_multispecies_fba().systems[1].elements]
        ['biomass_fast', 'biomass_slow', 'glucose']
    """
    cells = []
    interactions = []
    for name, vmax in (('fast', fast_vmax), ('slow', slow_vmax)):
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
                km=km,
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
        systems=[
            System('fba', elements=cells, time=DiscreteTime()),
            System(
                'culture',
                elements=[
                    BiomassElement('biomass_fast', x=biomass),
                    BiomassElement('biomass_slow', x=biomass),
                    SubstrateElement('glucose', c=glucose),
                ],
            ),
        ],
        interactions=interactions,
    )
