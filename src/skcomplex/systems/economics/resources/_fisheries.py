"""Fishery systems: the life cycle of a migratory fish through stages.

Natural losses and the catch deplete the stages.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from functools import partial
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import TransferInteraction
from ....core.components.laws import Sink, Source
from ....core.components.laws.utils import minimum
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec

TONNES_PER_MILLION_KG = 1e3
FEMALE_FRACTION = 0.5
MAX_NESTS = 60.0
SPAWNING_TIME = 1.0
STAGES = (
    ('eggs', 'hatching_time', 1.0, 'egg_loss', 0.99),
    ('larvae', 'larval_time', 2.0, 'larval_loss', 0.8),
    ('juveniles', 'river_time', 8.0, 'juvenile_loss', 0.4),
    ('yearlings', 'ocean_time', 12.0, 'first_year_loss', 0.3),
    ('adults', 'adult_ocean_time', 12.0, 'second_year_loss', 0.5),
    ('migrants', 'migration_time', 2.0, 'migration_loss', 0.25),
)


def _stage(
    label: str,
    stock: str,
    following: str,
    time: str,
    losses: tuple[tuple[str, str], ...],
) -> TransferInteraction:
    """Return the transfer that drains a stage over its duration.

    A share goes to each loss and the rest to the next stage. The stage reads the duration `time` and the share of every
    loss by name, and reports the flow to the next stage as `hilsa.<label>` and the flow to every loss as
    `hilsa.<label>_<loss>`.
    """
    shares = [share for share, _ in losses]
    return TransferInteraction(
        label,
        source='hilsa',
        target='hilsa',
        state=stock,
        into=following,
        amount=partial(_survival, stock=stock, time=time, shares=shares),
        flow=label,
        legs=[
            (stock, Sink(sink), partial(_loss, stock=stock, time=time, share=share), f'{label}_{sink}')
            for share, sink in losses
        ],
    )


def _outflow(s: SimpleNamespace, stock: str, time: str) -> Expression:
    """Return the outflow of a stage over its duration."""
    return getattr(s.source, stock) / getattr(s.source, time)


def _survival(s: SimpleNamespace, stock: str, time: str, shares: list[str]) -> Expression:
    """Return the outflow of a stage that survives its losses into the next stage."""
    return (1 - sum((getattr(s.source, share) for share in shares), start=0)) * _outflow(s, stock, time)


def _loss(s: SimpleNamespace, stock: str, time: str, share: str) -> Expression:
    """Return the outflow of a stage to one loss."""
    return getattr(s.source, share) * _outflow(s, stock, time)


def _spawning(s: SimpleNamespace) -> Expression:
    """Return the spawning.

    It is the adults about to spawn that the catch spares over the spawning time.
    """
    return (1 - s.adult_harvest) * s.spawners / s.spawning_time


def _nests(s: SimpleNamespace) -> Expression:
    """Return the nests of the spawners, limited by the spawning grounds."""
    return minimum(s.female_fraction * _spawning(s), s.max_nests)


def _adult_weight(s: SimpleNamespace) -> Expression:
    """Return the weight of the adults about to spawn, in tonnes."""
    return s.weight_per_adult * s.spawners * TONNES_PER_MILLION_KG


def _juvenile_weight(s: SimpleNamespace) -> Expression:
    """Return the weight of the juveniles, in tonnes."""
    return s.weight_per_juvenile * s.juveniles * TONNES_PER_MILLION_KG


class HilsaElement(Element):
    """Life cycle of the hilsa, with one stock of fish per stage.

    It holds the parameters of the stages and the catch.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        eggs:
            The eggs in the nests, in millions of fish.

        larvae:
            The larvae in the rivers, in millions of fish.

        juveniles:
            The juveniles in the rivers, in millions of fish.

        yearlings:
            The fish in their first year in the ocean, in millions of fish.

        adults:
            The fish in their second year in the ocean, in millions of fish.

        migrants:
            The adults migrating back to the rivers, in millions of fish.

        spawners:
            The adults about to spawn, in millions of fish.

        female_fraction:
            The fraction of the spawners that are female and build a nest.

        eggs_per_nest:
            The number of eggs per nest.

        max_nests:
            The most nests the spawning grounds hold.

        spawning_time:
            The duration of the spawning, in months.

        hatching_time:
            The time the eggs take to hatch, in months.

        larval_time:
            The time the larvae take to become juveniles, in months.

        river_time:
            The time the juveniles stay in the rivers, in months.

        ocean_time:
            The time the fish spend in their first year in the ocean, in months.

        adult_ocean_time:
            The time the fish spend in their second year in the ocean, in months.

        migration_time:
            The time the migration back to the rivers takes, in months.

        egg_loss:
            The fraction of the eggs lost to natural mortality.

        larval_loss:
            The fraction of the larvae lost to natural mortality.

        juvenile_loss:
            The fraction of the juveniles lost to natural mortality.

        first_year_loss:
            The fraction of the fish lost to natural mortality in their first year in the ocean.

        second_year_loss:
            The fraction of the fish lost to natural mortality in their second year in the ocean.

        migration_loss:
            The fraction of the adults lost during their migration to the rivers.

        juvenile_harvest:
            The fraction of the juveniles leaving the rivers that the fishery catches.

        adult_harvest:
            The fraction of the adults about to spawn that the fishery catches.

        weight_per_adult:
            The weight of an adult, in kilograms.

        weight_per_juvenile:
            The weight of a juvenile, in kilograms.
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        eggs: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 0.0,
        larvae: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 0.0,
        juveniles: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 0.0,
        yearlings: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 0.0,
        adults: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 0.0,
        migrants: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 0.0,
        spawners: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo), quantity='fish')] = 375.0,
        female_fraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = FEMALE_FRACTION,
        eggs_per_nest: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2e5,
        max_nests: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = MAX_NESTS,
        spawning_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = SPAWNING_TIME,
        hatching_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        larval_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        river_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 8.0,
        ocean_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12.0,
        adult_ocean_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 12.0,
        migration_time: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 2.0,
        egg_loss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.99,
        larval_loss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        juvenile_loss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.4,
        first_year_loss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        second_year_loss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        migration_loss: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.25,
        juvenile_harvest: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.48,
        adult_harvest: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
        weight_per_adult: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.82,
        weight_per_juvenile: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.09,
    ) -> None:
        super().__init__(label, visual=visual)
        self.eggs = eggs
        self.larvae = larvae
        self.juveniles = juveniles
        self.yearlings = yearlings
        self.adults = adults
        self.migrants = migrants
        self.spawners = spawners
        self.female_fraction = female_fraction
        self.eggs_per_nest = eggs_per_nest
        self.max_nests = max_nests
        self.spawning_time = spawning_time
        self.hatching_time = hatching_time
        self.larval_time = larval_time
        self.river_time = river_time
        self.ocean_time = ocean_time
        self.adult_ocean_time = adult_ocean_time
        self.migration_time = migration_time
        self.egg_loss = egg_loss
        self.larval_loss = larval_loss
        self.juvenile_loss = juvenile_loss
        self.first_year_loss = first_year_loss
        self.second_year_loss = second_year_loss
        self.migration_loss = migration_loss
        self.juvenile_harvest = juvenile_harvest
        self.adult_harvest = adult_harvest
        self.weight_per_adult = weight_per_adult
        self.weight_per_juvenile = weight_per_juvenile


class HilsaFisherySystem(BaseSystem):
    """A hilsa fishery, the life cycle of a migratory fish.

    The catch of juveniles and spawners depletes it. Spawning adults lay eggs in nests in the rivers, and the number of
    nests is limited by the spawning grounds. The eggs hatch into larvae within a month, the larvae become juveniles
    within two months, and the juveniles stay in the rivers for eight months before they enter the ocean. The fish
    mature in the ocean over two years, migrate back to the rivers, and spawn. Every stage drains over its duration, and
    a fixed fraction of its outflow is lost to natural mortality, to the catch of juveniles, or to the catch of spawning
    adults, while the rest enters the next stage. Fish are conserved, every change of a stage is a flow, and the
    population settles where the spawning grounds fill, unless the catch keeps the reproduction below replacement. The
    stages, the durations, and the loss fractions follow the hilsa system of Bala, Arshad and Noh (2017, Chapter 9),
    with the stocks in millions of fish and time in months.

    The system has the stages `hilsa.eggs`, `hilsa.larvae`, `hilsa.juveniles`, `hilsa.yearlings`, `hilsa.adults`,
    `hilsa.migrants`, and `hilsa.spawners`, and the flows between the stages, among them `hilsa.egg_deposition`,
    `hilsa.young_entering_ocean_catch`, and `hilsa.adult_catch`, per month, over months. The observables of the system
    are the nests `hilsa.nests`, and the weights `hilsa.adult_weight` and `hilsa.juvenile_weight` in tonnes.

    Args:
        spawners:
            The initial number of adults about to spawn.

        eggs_per_nest:
            The number of eggs per nest.

        juvenile_harvest:
            The fraction of the juveniles leaving the rivers that the fishery catches.

        adult_harvest:
            The fraction of the adults about to spawn that the fishery catches.

        migration_loss:
            The fraction of the adults lost during their migration to the rivers.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import HilsaFisherySystem
        >>> HilsaFisherySystem(adult_harvest=0.5).prepare().system_.elements[0].adult_harvest
        0.5
    """

    field = 'economics/resources'
    topics = ('fishery', 'life cycle', 'stocks and flows', 'system dynamics', 'ode')
    references = ('Bala et al. 2017',)
    until = 120.0
    dt = 0.5
    observables = (
        ObservableSpec('nests', _nests, element='hilsa'),
        ObservableSpec('adult_weight', _adult_weight, element='hilsa'),
        ObservableSpec('juvenile_weight', _juvenile_weight, element='hilsa'),
    )

    def __init__(
        self,
        spawners: float = 375.0,
        *,
        eggs_per_nest: float = 2e5,
        juvenile_harvest: float = 0.48,
        adult_harvest: float = 0.8,
        migration_loss: float = 0.25,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.spawners = spawners
        self.eggs_per_nest = eggs_per_nest
        self.juvenile_harvest = juvenile_harvest
        self.adult_harvest = adult_harvest
        self.migration_loss = migration_loss

    @staticmethod
    def _birth_death() -> list[TransferInteraction]:
        """Return the life cycle flows of the stages of the hilsa.

        They are the spawning, hatching, growth, migration, losses, and catch.
        """
        mortality = 'mortality'
        return [
            TransferInteraction(
                'egg_deposition',
                source='hilsa',
                target='hilsa',
                state=Source('spawning'),
                into='eggs',
                amount=lambda s: s.source.eggs_per_nest * _nests(s.source),
                flow='egg_deposition',
            ),
            _stage('hatching', 'eggs', 'larvae', 'hatching_time', (('egg_loss', mortality),)),
            _stage('juvenile_entering', 'larvae', 'juveniles', 'larval_time', (('larval_loss', mortality),)),
            _stage(
                'young_entering_ocean',
                'juveniles',
                'yearlings',
                'river_time',
                (('juvenile_loss', mortality), ('juvenile_harvest', 'catch')),
            ),
            _stage('maturation', 'yearlings', 'adults', 'ocean_time', (('first_year_loss', mortality),)),
            _stage('adults_entering', 'adults', 'migrants', 'adult_ocean_time', (('second_year_loss', mortality),)),
            _stage('adults_leaving', 'migrants', 'spawners', 'migration_time', (('migration_loss', mortality),)),
            TransferInteraction(
                'spawning',
                source='hilsa',
                target='hilsa',
                state='spawners',
                into=Sink('spent'),
                amount=lambda s: _spawning(s.source),
                flow='spawning',
                legs=[
                    (
                        'spawners',
                        Sink('catch'),
                        lambda s: s.source.adult_harvest * s.source.spawners / s.source.spawning_time,
                        'adult_catch',
                    ),
                ],
            ),
        ]

    @staticmethod
    def _filled_stages(
        spawners: float,
        eggs_per_nest: float,
        juvenile_harvest: float,
        adult_harvest: float,
        migration_loss: float,
    ) -> dict[str, float]:
        """Return the stages filled by the egg deposition of the initial spawners.

        The deposition fills them through the life cycle.
        """
        spawning = FEMALE_FRACTION * (1 - adult_harvest) * spawners / SPAWNING_TIME
        inflow = eggs_per_nest * min(spawning, MAX_NESTS)
        losses = {'juveniles': STAGES[2][4] + juvenile_harvest, 'migrants': migration_loss}
        stages = {}
        for name, _, duration, _, fraction in STAGES:
            stages[name] = inflow * duration
            inflow *= 1 - losses.get(name, fraction)
        return {**stages, 'spawners': spawners}

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        hilsa = HilsaElement(
            'hilsa',
            eggs_per_nest=self.eggs_per_nest,
            juvenile_harvest=self.juvenile_harvest,
            adult_harvest=self.adult_harvest,
            migration_loss=self.migration_loss,
        )
        hilsa.set_params(
            **self._filled_stages(
                self.spawners,
                self.eggs_per_nest,
                self.juvenile_harvest,
                self.adult_harvest,
                self.migration_loss,
            ),
        )
        return CompositeSystem(self.label, elements=[hilsa], interactions=self._birth_death())
