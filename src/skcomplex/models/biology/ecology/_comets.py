"""COMETS: dynamic flux balance analysis on a lattice whose biomass and glucose diffuse."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np

from ....core.elements import FieldElement, GroupElement
from ....core.interactions import DiffusionInteraction
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.solvers import RK4Solver
from ....specs import VisualSpec
from ..._catalog import register_model
from ..common.metabolism import (
    BiomassElement,
    ExchangeInteraction,
    GrowthInteraction,
    SubstrateElement,
    load_network,
    metabolic_network_type,
)

GROWTH = 'Biomass_Ecoli_core'
GLUCOSE_EXCHANGE = 'EX_glc__D_e'
STEP = 0.05


@register_model(
    field='biology/ecology',
    tags=('COMETS', 'dynamic flux balance', 'lattice', 'diffusion', 'linear program', 'metabolism'),
    references=('Raman 2021', 'Harcombe et al. 2014'),
    defaults={'until': 5.0, 'dt': 0.1, 'solver': RK4Solver(dt=STEP)},
)
def comets(
    shape: tuple[int, int] = (25, 25),
    *,
    inoculum: float = 0.1,
    glucose: float = 5.0,
    biomass_diffusion: float = 0.2,
    glucose_diffusion: float = 1.0,
    threshold: float = 1e-3,
    vmax: float = 10.0,
    km: float = 0.015,
) -> System:
    """Return an expanding colony of the E. coli core model on a lattice, which needs the extra `metabolism`.

    Every site holds a copy of the program, which solves only where the biomass lies above a threshold, with the uptake
    bounded by the glucose of the site (Raman 2021, Section 11.1.3.4). Biomass and glucose diffuse between the sites, so
    a colony started at the centre spreads at a nearly constant speed. The program solves at every evaluation of the
    drift, by the direct approach with fixed steps.

    Args:
        shape:
            The number of sites along each side.

        inoculum:
            The initial biomass at the central site.

        glucose:
            The initial glucose of every site.

        biomass_diffusion:
            The diffusion coefficient of the biomass.

        glucose_diffusion:
            The diffusion coefficient of the glucose.

        threshold:
            The biomass above which a site solves its program.

        vmax:
            The largest glucose uptake.

        km:
            The half-saturation concentration of glucose.

    Returns:
        system:
            The system.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.models.biology.ecology import comets
        >>> [element.label for element in comets(shape=(2, 2)).elements]
        ['cells', 'biomass', 'glucose']
    """
    source = load_network('textbook', 'cell', needed_by='comets')
    plain = {name: replace(spec, rule='') for name, spec in source.reactions.items()}
    cell_type = metabolic_network_type('ColonyCellElement', plain, source.default_objective)
    initial = np.zeros(shape)
    initial[shape[0] // 2, shape[1] // 2] = inoculum
    space = LatticeSpace(shape)
    images = VisualSpec('image', {'states': [cell_type.flux_name(GROWTH)]})
    return System(
        elements=[
            GroupElement('cells', element=cell_type, on='sites', space=space, visual=images),
            FieldElement('biomass', element=BiomassElement, space=space, values={'x': initial}),
            FieldElement(
                'glucose',
                element=SubstrateElement,
                space=space,
                values={'c': np.full(shape, float(glucose))},
            ),
        ],
        interactions=[
            ExchangeInteraction(
                'uptake',
                cell='cells',
                biomass='biomass',
                substrate='glucose',
                reaction=GLUCOSE_EXCHANGE,
                vmax=vmax,
                km=km,
                threshold=threshold,
            ),
            GrowthInteraction('growth', cell='cells', biomass='biomass', reaction=GROWTH),
            DiffusionInteraction('biomass_spread', target='biomass', state='x', coefficient=biomass_diffusion),
            DiffusionInteraction('glucose_spread', target='glucose', state='c', coefficient=glucose_diffusion),
        ],
        space=space,
    )
