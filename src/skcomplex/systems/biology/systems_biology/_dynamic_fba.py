"""Dynamic flux balance analysis of a batch culture of the E. coli core system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping

from ....core.components.spaces import DiscreteTime
from ....core.components.system import System
from ....core.engine.solvers import CoSimulationSolver, RK4Solver
from ....core.errors import ValidationError
from ..._catalog import register_system
from ..common.metabolism import (
    BiomassElement,
    ExchangeInteraction,
    GrowthInteraction,
    SubstrateElement,
    load_network,
)

APPROACHES = ('static', 'direct')
GROWTH = 'Biomass_Ecoli_core'
GLUCOSE_EXCHANGE = 'EX_glc__D_e'
SYNCHRONISATION = 0.05
ODE_STEP = 0.005


def _settings(*, approach: str = 'static', interval: float = SYNCHRONISATION, **_: object) -> Mapping[str, object]:
    """Return the run settings of an approach and a synchronisation interval."""
    solver = (
        CoSimulationSolver(sync_dt=interval, solvers={'culture': RK4Solver(dt=ODE_STEP)})
        if approach == 'static'
        else RK4Solver(dt=ODE_STEP)
    )
    return {'until': 10.0, 'dt': SYNCHRONISATION, 'solver': solver}


@register_system(
    field='biology/systems_biology',
    tags=('dynamic flux balance', 'linear program', 'metabolism', 'batch culture', 'co-simulation'),
    references=('Raman 2021', 'Mahadevan et al. 2002', 'Varma and Palsson 1994'),
    defaults={'until': 10.0, 'dt': SYNCHRONISATION},
    settings=_settings,
)
def dynamic_fba(
    approach: str = 'static',
    *,
    biomass: float = 0.01,
    glucose: float = 10.0,
    vmax: float = 10.0,
    km: float = 0.015,
    interval: float = SYNCHRONISATION,
) -> System:
    """Return a batch culture of the E. coli core system, which needs the extra `metabolism`.

    The program solves for the fluxes with Michaelis-Menten glucose uptake and then the smallest total flux. By the
    static optimisation approach it solves in a discrete partition at every synchronisation point, and the biomass and
    the glucose integrate with the fluxes held (Raman 2021, Eqs. 9.8 and 9.9). By the direct approach it solves at every
    evaluation of the drift. Without glucose the program is infeasible, and the policy `'zero'` stops growth, so biomass
    grows in one phase and stays.

    Args:
        approach:
            `'static'` for the static optimisation approach or `'direct'` for the direct approach.

        biomass:
            The initial biomass in gDW per L.

        glucose:
            The initial glucose in mM.

        vmax:
            The largest glucose uptake in mmol per gDW per h.

        km:
            The half-saturation concentration of glucose in mM.

        interval:
            The synchronisation interval of the static optimisation approach in h.

    Returns:
        system:
            The system.

    Raises:
        ValidationError:
            The approach is unknown.

        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.systems_biology import dynamic_fba
        >>> [system.label for system in dynamic_fba().systems]
        ['fba', 'culture']
    """
    if approach not in APPROACHES:
        error_msg = f'Unknown approach {approach!r}. Use one of {", ".join(APPROACHES)}.'
        raise ValidationError(error_msg)
    cell = load_network('textbook', 'cell', needed_by='dynamic_fba')
    cell.parsimonious, cell.infeasible = True, 'zero'
    amounts = [BiomassElement('biomass', x=biomass), SubstrateElement('glucose', c=glucose)]
    static = approach == 'static'
    cell_path = 'fba/cell' if static else 'cell'
    biomass_path = 'culture/biomass' if static else 'biomass'
    interactions = [
        ExchangeInteraction(
            'uptake',
            cell=cell_path,
            biomass=biomass_path,
            substrate='culture/glucose' if static else 'glucose',
            reaction=GLUCOSE_EXCHANGE,
            vmax=vmax,
            km=km,
            interval=interval if static else None,
        ),
        GrowthInteraction('growth', cell=cell_path, biomass=biomass_path, reaction=GROWTH),
    ]
    if static:
        return System(
            systems=[System('fba', elements=[cell], time=DiscreteTime()), System('culture', elements=amounts)],
            interactions=interactions,
        )
    return System(elements=[cell, *amounts], interactions=interactions)
