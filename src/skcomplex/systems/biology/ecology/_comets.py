"""COMETS: growth by a linear program at every site of a lattice, alternating with diffusion."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np

from ....core.components.elements import FieldElement, GroupElement
from ....core.components.interactions import DiffusionInteraction
from ....core.components.spaces import LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import IMEXEulerSolver
from ....core.specs import VisualSpec
from ..common.metabolism import (
    BiomassElement,
    ExchangeInteraction,
    GrowthInteraction,
    MetabolicNetworkElement,
    SubstrateElement,
    load_network,
)

GROWTH = 'Biomass_Ecoli_core'
GLUCOSE_EXCHANGE = 'EX_glc__D_e'
STEP = 0.1
SECONDS_PER_HOUR = 3600.0
MMOL_PER_MOL = 1000.0
GLUCOSE_MOLAR_MASS = 180.16


class CometsSystem(BaseSystem):
    """An expanding colony of the E. coli core system on a lattice, which needs the extra `metabolism`.

    Run with `IMEXEulerSolver`, every step is a growth step and then a diffusion step, the two steps that COMETS
    alternates (Raman 2021, Section 11.1.3.4). The growth step solves the program of every site that holds biomass and
    applies the update rules of Harcombe et al. (2014). The diffusion step solves one implicit Euler system, where
    Harcombe et al. use an alternating direction implicit scheme. Biomass and glucose both diffuse, as in Harcombe et
    al., while the short description of Raman names metabolites only.

    The states are gDW per L and mM of the medium under a box. A site whose program has no solution has zero fluxes,
    which happens when its glucose cannot pay for the maintenance flux of the network.

    Harcombe et al. in what follows is the main text of Harcombe et al. (2014), and the source of each default is named
    under its argument. The time step of 0.1 h of the registered run is from its In silico experiments. The length of
    the registered run keeps it within the budget of the gallery and has no source.

    These choices have no source in that text. The periodic edges are those of `LatticeSpace`, since the text does not
    state the outer boundary. The volume under a box is the amount of glucose under it over the concentration of the
    medium, since the text defers the volume to its supplement, which was not read. The molar mass of glucose is 180.16
    g/mol.

    The network is the E. coli core system, where Harcombe et al. use iJO1366. The death rate of 1% per step, the
    largest colony height of 200 um, and the oxygen of 0.25 mmol/cm^2 of Harcombe et al. are left out, so the biomass of
    a box has no upper limit and oxygen is limited only by the network.

    Args:
        shape:
            The number of boxes along each side, by default from Harcombe et al., Colony expansion comparisons.

        box:
            The side of a box in cm, by default from Harcombe et al., Colony expansion comparisons.

        inoculum:
            The initial biomass of the central box in g, by default from Harcombe et al., Colony expansion comparisons.

        glucose:
            The glucose of the medium in g/L, by default from Harcombe et al., In silico experiments.

        plate_glucose:
            The glucose of the medium per area of the plate in mmol/cm^2, by default from Harcombe et al., In silico
            experiments.

        biomass_diffusion:
            The diffusion coefficient of the biomass in cm^2/s, by default the value for colonies from Harcombe et al.,
            Table 1.

        glucose_diffusion:
            The diffusion coefficient of the glucose in cm^2/s, by default from Harcombe et al., Table 1.

        vmax:
            The largest glucose uptake in mmol/g/h, by default from Harcombe et al., Table 1.

        km:
            The half-saturation concentration of glucose in mM, by default from Harcombe et al., Table 1.

        interval:
            The time in h over which the glucose of a site must last, by default the time step, the bound of Raman
            (2021, Section 9.6).

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        MissingExtraError:
            The extra is not installed.

    Examples:
        >>> from skcomplex.systems.biology.ecology import CometsSystem
        >>> [element.label for element in CometsSystem(shape=(2, 2)).prepare().system_.elements]
        ['cells', 'biomass', 'glucose']
    """

    field = 'biology/ecology'
    topics = ('COMETS', 'dynamic flux balance', 'lattice', 'diffusion', 'linear program', 'metabolism')
    references = ('Raman 2021', 'Harcombe et al. 2014')
    until = 14.0
    dt = STEP
    solver = IMEXEulerSolver(dt=STEP)

    def __init__(
        self,
        shape: tuple[int, int] = (50, 50),
        *,
        box: float = 0.02,
        inoculum: float = 3e-7,
        glucose: float = 5.0,
        plate_glucose: float = 0.0088,
        biomass_diffusion: float = 3e-10,
        glucose_diffusion: float = 5e-6,
        vmax: float = 10.0,
        km: float = 0.01,
        interval: float = STEP,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.shape = shape
        self.box = box
        self.inoculum = inoculum
        self.glucose = glucose
        self.plate_glucose = plate_glucose
        self.biomass_diffusion = biomass_diffusion
        self.glucose_diffusion = glucose_diffusion
        self.vmax = vmax
        self.km = km
        self.interval = interval

    def _build_system(self) -> System:
        """Return the parts of an expanding colony of the E. coli core system on a lattice."""
        source = load_network('textbook', 'cell', needed_by='comets')
        plain = {name: replace(spec, rule='') for name, spec in source.reactions.items()}
        concentration = self.glucose / GLUCOSE_MOLAR_MASS * MMOL_PER_MOL
        volume = self.plate_glucose * self.box**2 / concentration
        initial = np.zeros(self.shape)
        initial[self.shape[0] // 2, self.shape[1] // 2] = self.inoculum / volume
        space = LatticeSpace(self.shape, spacing=self.box)
        images = VisualSpec('image', {'states': [MetabolicNetworkElement.flux_name(GROWTH)]})
        cells = {
            'name': 'ColonyCell',
            'reactions': plain,
            'default_objective': source.default_objective,
            'infeasible': 'zero',
        }
        return System(
            self.label,
            elements=[
                GroupElement(
                    'cells',
                    element=MetabolicNetworkElement,
                    values=cells,
                    on='sites',
                    space=space,
                    visual=images,
                ),
                FieldElement('biomass', element=BiomassElement, space=space, values={'x': initial}),
                FieldElement(
                    'glucose',
                    element=SubstrateElement,
                    space=space,
                    values={'c': np.full(self.shape, concentration)},
                ),
            ],
            interactions=[
                ExchangeInteraction(
                    'uptake',
                    cell='cells',
                    biomass='biomass',
                    substrate='glucose',
                    reaction=GLUCOSE_EXCHANGE,
                    vmax=self.vmax,
                    km=self.km,
                    interval=self.interval,
                    threshold=0.0,
                ),
                GrowthInteraction('growth', cell='cells', biomass='biomass', reaction=GROWTH),
                DiffusionInteraction(
                    'biomass_spread',
                    target='biomass',
                    state='x',
                    coefficient=self.biomass_diffusion * SECONDS_PER_HOUR,
                ),
                DiffusionInteraction(
                    'glucose_spread',
                    target='glucose',
                    state='c',
                    coefficient=self.glucose_diffusion * SECONDS_PER_HOUR,
                ),
            ],
            space=space,
        )
