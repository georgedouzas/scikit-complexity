"""Fishery models: the life cycle of a migratory fish through stages that natural losses and the catch deplete."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, Expression, parameter, state
from ....core.components import Element
from ....core.laws import Auxiliary, Law, Sink, Source, Symbols, Transfer, where
from ....core.runtime import register_model
from ....core.system import System

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
) -> list[Law]:
    """Return the flows that drain a stage over its duration, a share to each loss and the rest to the next stage."""
    outflow = stock / time
    survival = 1 - sum((share for share, _ in losses), start=0)
    return [
        Transfer(stock, following, survival * outflow, label=label),
        *(Transfer(stock, Sink(sink), share * outflow, label=f'{label}_{sink}') for share, sink in losses),
    ]


def _hilsa_laws(s: Symbols) -> list[Law]:
    """Return the spawning, hatching, growth, migration, losses, and catch of the stages of the hilsa life cycle."""
    spawning = (1 - s.adult_harvest) * s.spawners / s.spawning_time
    nests = Auxiliary('nests', _smaller(s.female_fraction * spawning, s.max_nests))
    adult_weight = Auxiliary('adult_weight', s.weight_per_adult * s.spawners * TONNES_PER_MILLION_KG)
    juvenile_weight = Auxiliary('juvenile_weight', s.weight_per_juvenile * s.juveniles * TONNES_PER_MILLION_KG)
    return [
        nests,
        adult_weight,
        juvenile_weight,
        Transfer(Source('spawning'), s.eggs, s.eggs_per_nest * nests.symbol, label='egg_deposition'),
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
        Transfer(s.spawners, Sink('spent'), spawning, label='spawning'),
        Transfer(s.spawners, Sink('catch'), s.adult_harvest * s.spawners / s.spawning_time, label='adult_catch'),
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


Hilsa = Element.define(
    'Hilsa',
    states={
        'eggs': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='fish'),
        'larvae': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='fish'),
        'juveniles': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='fish'),
        'yearlings': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='fish'),
        'adults': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='fish'),
        'migrants': state(default=0.0, domain=Domain.NONNEGATIVE, quantity='fish'),
        'spawners': state(default=375.0, domain=Domain.NONNEGATIVE, quantity='fish'),
    },
    parameters={
        'female_fraction': parameter(default=FEMALE_FRACTION, domain=Domain.NONNEGATIVE),
        'eggs_per_nest': parameter(default=2e5, domain=Domain.NONNEGATIVE),
        'max_nests': parameter(default=MAX_NESTS, domain=Domain.NONNEGATIVE),
        'spawning_time': parameter(default=SPAWNING_TIME, domain=Domain.POSITIVE),
        **{time: parameter(default=duration, domain=Domain.POSITIVE) for _, time, duration, _, _ in STAGES},
        **{loss: parameter(default=fraction, domain=Domain.NONNEGATIVE) for _, _, _, loss, fraction in STAGES},
        'juvenile_harvest': parameter(default=0.48, domain=Domain.NONNEGATIVE),
        'adult_harvest': parameter(default=0.8, domain=Domain.NONNEGATIVE),
        'weight_per_adult': parameter(default=0.82, domain=Domain.NONNEGATIVE),
        'weight_per_juvenile': parameter(default=0.09, domain=Domain.NONNEGATIVE),
    },
    laws=_hilsa_laws,
)


@register_model(
    field='economics/resources',
    tags=('fishery', 'life cycle', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Bala et al. 2017',),
    defaults={'until': 120.0, 'dt': 0.5},
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
            `hilsa.adults`, `hilsa.migrants`, and `hilsa.spawners`, the auxiliaries `hilsa.nests`,
            `hilsa.adult_weight`, and `hilsa.juvenile_weight` in tonnes, and the flows between the stages, among them
            `hilsa.egg_deposition`, `hilsa.young_entering_ocean_catch`, and `hilsa.adult_catch`, per month, over
            months.

    Examples:
        >>> from skcomplex.models.economics.resources import hilsa_fishery
        >>> hilsa_fishery(adult_harvest=0.5).elements[0].adult_harvest
        0.5
    """
    hilsa = Hilsa(
        'hilsa',
        eggs_per_nest=eggs_per_nest,
        juvenile_harvest=juvenile_harvest,
        adult_harvest=adult_harvest,
        migration_loss=migration_loss,
    )
    hilsa.set_params(**_filled_stages(spawners, eggs_per_nest, juvenile_harvest, adult_harvest, migration_loss))
    return System(components=[hilsa])
