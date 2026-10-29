"""Fishery models: the life cycle of a migratory fish through stages that natural losses and the catch deplete."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, Sink, Source, TransferLaw
from ....core.laws.utils import where
from ....core.system import System
from ....specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model

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


def _smaller(first: Expression, second: Expression) -> Expression:
    """Return the smaller of two symbolic values."""
    return where(first < second, first, second)


def _stage(
    stock: Expression,
    following: Expression | Sink,
    time: Expression,
    losses: list[tuple[Expression, str]],
    label: str,
) -> list[BaseLaw]:
    """Return the flows that drain a stage over its duration, a share to each loss and the rest to the next stage."""
    outflow = stock / time
    survival = 1 - sum((share for share, _ in losses), start=0)
    return [
        TransferLaw(stock, following, survival * outflow, label=label),
        *(TransferLaw(stock, Sink(sink), share * outflow, label=f'{label}_{sink}') for share, sink in losses),
    ]


def _spawning(s: SimpleNamespace) -> Expression:
    """Return the spawning, the adults about to spawn that the catch spares over the spawning time."""
    return (1 - s.adult_harvest) * s.spawners / s.spawning_time


def _nests(s: SimpleNamespace) -> Expression:
    """Return the nests of the spawners, limited by the spawning grounds."""
    return _smaller(s.female_fraction * _spawning(s), s.max_nests)


def _adult_weight(s: SimpleNamespace) -> Expression:
    """Return the weight of the adults about to spawn, in tonnes."""
    return s.weight_per_adult * s.spawners * TONNES_PER_MILLION_KG


def _juvenile_weight(s: SimpleNamespace) -> Expression:
    """Return the weight of the juveniles, in tonnes."""
    return s.weight_per_juvenile * s.juveniles * TONNES_PER_MILLION_KG


def _hilsa_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the spawning, hatching, growth, migration, losses, and catch of the stages of the hilsa life cycle."""
    spawning = _spawning(s)
    return [
        TransferLaw(Source('spawning'), s.eggs, s.eggs_per_nest * _nests(s), label='egg_deposition'),
        *_stage(s.eggs, s.larvae, s.hatching_time, [(s.egg_loss, 'mortality')], 'hatching'),
        *_stage(s.larvae, s.juveniles, s.larval_time, [(s.larval_loss, 'mortality')], 'juvenile_entering'),
        *_stage(
            s.juveniles,
            s.yearlings,
            s.river_time,
            [(s.juvenile_loss, 'mortality'), (s.juvenile_harvest, 'catch')],
            'young_entering_ocean',
        ),
        *_stage(s.yearlings, s.adults, s.ocean_time, [(s.first_year_loss, 'mortality')], 'maturation'),
        *_stage(s.adults, s.migrants, s.adult_ocean_time, [(s.second_year_loss, 'mortality')], 'adults_entering'),
        *_stage(s.migrants, s.spawners, s.migration_time, [(s.migration_loss, 'mortality')], 'adults_leaving'),
        TransferLaw(s.spawners, Sink('spent'), spawning, label='spawning'),
        TransferLaw(s.spawners, Sink('catch'), s.adult_harvest * s.spawners / s.spawning_time, label='adult_catch'),
    ]


def _filled_stages(
    spawners: float,
    eggs_per_nest: float,
    juvenile_harvest: float,
    adult_harvest: float,
    migration_loss: float,
) -> dict[str, float]:
    """Return the stages that the egg deposition of the initial spawners fills as it passes through the life cycle."""
    spawning = FEMALE_FRACTION * (1 - adult_harvest) * spawners / SPAWNING_TIME
    inflow = eggs_per_nest * min(spawning, MAX_NESTS)
    losses = {'juveniles': STAGES[2][4] + juvenile_harvest, 'migrants': migration_loss}
    stages = {}
    for name, _, duration, _, fraction in STAGES:
        stages[name] = inflow * duration
        inflow *= 1 - losses.get(name, fraction)
    return {**stages, 'spawners': spawners}


class HilsaElement(Element):
    """Life cycle of the hilsa, with one stock of fish per stage and the parameters of the stages and the catch.

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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the flows of the life cycle.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return _hilsa_laws(s)


@register_model(
    field='economics/resources',
    tags=('fishery', 'life cycle', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.5},
    observables=lambda system: [
        ObservableSpec('nests', _nests, element='hilsa'),
        ObservableSpec('adult_weight', _adult_weight, element='hilsa'),
        ObservableSpec('juvenile_weight', _juvenile_weight, element='hilsa'),
    ],
)
def hilsa_fishery(
    spawners: float = 375.0,
    *,
    eggs_per_nest: float = 2e5,
    juvenile_harvest: float = 0.48,
    adult_harvest: float = 0.8,
    migration_loss: float = 0.25,
) -> System:
    """Return a hilsa fishery, the life cycle of a migratory fish that the catch of juveniles and spawners depletes.

    Spawning adults lay eggs in nests in the rivers, and the number of nests is limited by the spawning grounds. The
    eggs hatch into larvae within a month, the larvae become juveniles within two months, and the juveniles stay in
    the rivers for eight months before they enter the ocean. The fish mature in the ocean over two years, migrate back
    to the rivers, and spawn. Every stage drains over its duration, and a fixed fraction of its outflow is lost to
    natural mortality, to the catch of juveniles, or to the catch of spawning adults, while the rest enters the next
    stage. Fish are conserved, every change of a stage is a flow, and the population settles where the spawning
    grounds fill, unless the catch keeps the reproduction below replacement. The stages, the durations, and the loss
    fractions follow the hilsa model of Bala, Arshad and Noh (2017, Chapter 9), with the stocks in millions of fish
    and time in months.

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

    Returns:
        system:
            The model, with the stages `hilsa.eggs`, `hilsa.larvae`, `hilsa.juveniles`, `hilsa.yearlings`,
            `hilsa.adults`, `hilsa.migrants`, and `hilsa.spawners`, and the flows between the stages, among them
            `hilsa.egg_deposition`, `hilsa.young_entering_ocean_catch`, and `hilsa.adult_catch`, per month, over
            months. The observables of the model are the nests `hilsa.nests`, and the weights `hilsa.adult_weight` and
            `hilsa.juvenile_weight` in tonnes.

    Examples:
        >>> from skcomplex.models.economics.resources import hilsa_fishery
        >>> hilsa_fishery(adult_harvest=0.5).elements[0].adult_harvest
        0.5
    """
    hilsa = HilsaElement(
        'hilsa',
        eggs_per_nest=eggs_per_nest,
        juvenile_harvest=juvenile_harvest,
        adult_harvest=adult_harvest,
        migration_loss=migration_loss,
    )
    hilsa.set_params(**_filled_stages(spawners, eggs_per_nest, juvenile_harvest, adult_harvest, migration_loss))
    return System(elements=[hilsa])
