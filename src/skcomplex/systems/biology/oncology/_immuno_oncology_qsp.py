"""A reduced quantitative systems pharmacology model of a tumour under anti-PD-1."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression, pi

from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ObservableSpec
from ...general.reactions import ReactionInteraction
from ..common.kinetics import CompartmentElement
from ..common.pharmacokinetics import DosingInteraction

CELL_VOLUME = 2.572e-9
CANCER = ('cancer', 'x')
EFFECTOR = ('effector', 'x')
TREG = ('treg', 'x')
CENTRAL = ('central', 'x')
PERIPHERAL = ('peripheral', 'x')
QSP_SOLVER = ScipyODESolver(method='LSODA', rtol=1e-8, atol=1e-6)


def _get_diameter(s: SimpleNamespace) -> Expression:
    """Return the diameter in centimetres of a sphere of the tumour volume."""
    diameter: Expression = (6 * CELL_VOLUME * s.cancer.x / pi) ** (1 / 3)
    return diameter


def _get_effector_treg_ratio(s: SimpleNamespace) -> Expression:
    """Return the effector T cells per regulatory T cell, with one cell added below."""
    ratio: Expression = s.effector.x / (s.treg.x + 1)
    return ratio


class ImmunoOncologyQspSystem(BaseSystem):
    """A reduced quantitative systems pharmacology model of a tumour under anti-PD-1.

    It follows the QSP-IO platform that Wang et al. (2024) review, with five compartments. Cancer cells `C` grow
    logistically at the rate `growth` up to `capacity`, and effector T cells `T` kill them at the rate
    `kill (1 - inhibition f) T C`, where `f = kd / (kd + c)` is the free share of PD-1 at the central antibody
    concentration `c`. PD-1 occupancy is at quasi steady state. The tumour recruits effector T cells at the rate
    `recruitment C` and regulatory T cells `R` at the rate `treg_recruitment C`. Effector T cells proliferate at the
    rate `proliferation T / (1 + R / suppression)`, which the regulatory T cells suppress, and both kinds die at their
    own rates. The antibody follows two-compartment kinetics in amounts, with the central amount `a1` and the
    peripheral amount `a2`, `da1/dt = dosing - (clearance + k12) a1 + k21 a2` and `da2/dt = k12 a1 - k21 a2`, and
    `c = a1 / volume`. Every dose enters over one day, every `period` days from `start`.

    Cells are counts, the antibody is in mg, the volume in litres, and time in days. The parameters are of the orders
    of the QSP-IO models of Jafarnejad et al. (2019) and Milberg et al. (2019), with nivolumab at 240 mg every two
    weeks. The tumour diameter in centimetres is that of a sphere of cells of 2572 cubic micrometres each. The system
    has the cells `cancer.x`, `effector.x`, and `treg.x`, the antibody `central.x` and `peripheral.x`, and the
    observables `diameter` and `effector_treg_ratio`.

    Args:
        cancer:
            The initial cancer cells.

        effector:
            The initial effector T cells.

        treg:
            The initial regulatory T cells.

        growth:
            The growth rate of the tumour, per day.

        capacity:
            The carrying capacity of the tumour, in cells.

        kill:
            The rate at which one effector T cell kills one cancer cell, per day.

        recruitment:
            The effector T cells recruited per cancer cell, per day.

        treg_recruitment:
            The regulatory T cells recruited per cancer cell, per day.

        proliferation:
            The proliferation rate of effector T cells, per day.

        suppression:
            The regulatory T cells that halve the proliferation of effector T cells.

        effector_death:
            The death rate of effector T cells, per day.

        treg_death:
            The death rate of regulatory T cells, per day.

        inhibition:
            The largest share of killing that free PD-1 blocks, between 0 and 1.

        kd:
            The antibody concentration that occupies half of PD-1, in mg/L.

        dose:
            The antibody of one dose, in mg, 0 for no treatment.

        period:
            The days between doses.

        start:
            The day of the first dose.

        volume:
            The volume of the central compartment, in litres.

        clearance:
            The elimination rate of the antibody from the central compartment, per day.

        k12:
            The transfer rate of the antibody from the central to the peripheral compartment, per day.

        k21:
            The transfer rate of the antibody from the peripheral to the central compartment, per day.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        Treatment shrinks the tumour of the default patient within six months.

        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.systems.biology.oncology import ImmunoOncologyQspSystem
        >>> treated = simulate(ImmunoOncologyQspSystem(), until=180.0, dt=1.0)
        >>> untreated = simulate(ImmunoOncologyQspSystem(dose=0.0), until=180.0, dt=1.0)
        >>> round(float(treated.get_values('diameter')[-1]), 2), round(float(untreated.get_values('diameter')[-1]), 2)
        (1.7, 3.85)
    """

    field = 'biology/oncology'
    topics = ('immunotherapy', 'ode', 'checkpoint inhibitor', 'PD-1', 'dosing', 'quantitative systems pharmacology')
    references = ('Wang et al. 2024', 'Jafarnejad et al. 2019', 'Milberg et al. 2019')
    until = 180.0
    dt = 1.0
    solver = QSP_SOLVER
    observables = (
        ObservableSpec('diameter', _get_diameter),
        ObservableSpec('effector_treg_ratio', _get_effector_treg_ratio),
    )

    def __init__(
        self,
        cancer: float = 4e9,
        effector: float = 1e8,
        treg: float = 5e7,
        *,
        growth: float = 0.02,
        capacity: float = 1e12,
        kill: float = 1e-10,
        recruitment: float = 0.01,
        treg_recruitment: float = 0.002,
        proliferation: float = 0.05,
        suppression: float = 1e9,
        effector_death: float = 0.1,
        treg_death: float = 0.1,
        inhibition: float = 0.9,
        kd: float = 0.4,
        dose: float = 240.0,
        period: float = 14.0,
        start: float = 0.0,
        volume: float = 4.0,
        clearance: float = 0.05,
        k12: float = 0.2,
        k21: float = 0.2,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.cancer = cancer
        self.effector = effector
        self.treg = treg
        self.growth = growth
        self.capacity = capacity
        self.kill = kill
        self.recruitment = recruitment
        self.treg_recruitment = treg_recruitment
        self.proliferation = proliferation
        self.suppression = suppression
        self.effector_death = effector_death
        self.treg_death = treg_death
        self.inhibition = inhibition
        self.kd = kd
        self.dose = dose
        self.period = period
        self.start = start
        self.volume = volume
        self.clearance = clearance
        self.k12 = k12
        self.k21 = k21

    def _build_system(self) -> CompositeSystem:
        """Return the cells and the antibody of the patient, and their reactions."""
        elements = [
            CompartmentElement('cancer', x=self.cancer),
            CompartmentElement('effector', x=self.effector),
            CompartmentElement('treg', x=self.treg),
            CompartmentElement('central'),
            CompartmentElement('peripheral'),
        ]

        def get_kill(s: SimpleNamespace) -> Expression:
            """Return the killing of cancer cells, which free PD-1 lowers."""
            cells, antibody = s.modifiers
            free = self.kd / (self.kd + antibody.x / self.volume)
            killing: Expression = self.kill * (1 - self.inhibition * free) * s.reactants[0].x * cells.x
            return killing

        def get_growth(s: SimpleNamespace) -> Expression:
            """Return the logistic growth of the tumour."""
            growth: Expression = self.growth * s.modifiers[0].x * (1 - s.modifiers[0].x / self.capacity)
            return growth

        def get_proliferation(s: SimpleNamespace) -> Expression:
            """Return the proliferation of effector T cells, which regulatory T cells suppress."""
            cells, tregs = s.modifiers
            proliferation: Expression = self.proliferation * cells.x / (1 + tregs.x / self.suppression)
            return proliferation

        interactions = [
            ReactionInteraction('growth', products={CANCER: 1}, modifiers={CANCER: 1}, propensity=get_growth),
            ReactionInteraction(
                'kill',
                reactants={CANCER: 1},
                modifiers={EFFECTOR: 1, CENTRAL: 1},
                propensity=get_kill,
            ),
            ReactionInteraction('recruitment', products={EFFECTOR: 1}, modifiers={CANCER: 1}, k=self.recruitment),
            ReactionInteraction(
                'proliferation',
                products={EFFECTOR: 1},
                modifiers={EFFECTOR: 1, TREG: 1},
                propensity=get_proliferation,
            ),
            ReactionInteraction('effector_death', reactants={EFFECTOR: 1}, k=self.effector_death),
            ReactionInteraction('treg_recruitment', products={TREG: 1}, modifiers={CANCER: 1}, k=self.treg_recruitment),
            ReactionInteraction('treg_death', reactants={TREG: 1}, k=self.treg_death),
            DosingInteraction('dosing', target='central', dose=self.dose, period=self.period, start=self.start),
            ReactionInteraction('distribution', reactants={CENTRAL: 1}, products={PERIPHERAL: 1}, k=self.k12),
            ReactionInteraction('redistribution', reactants={PERIPHERAL: 1}, products={CENTRAL: 1}, k=self.k21),
            ReactionInteraction('elimination', reactants={CENTRAL: 1}, k=self.clearance),
        ]
        return CompositeSystem(self.label, elements=elements, interactions=interactions)
