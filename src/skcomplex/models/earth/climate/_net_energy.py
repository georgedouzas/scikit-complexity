"""Net energy models: a finite fossil reserve whose energy return on investment falls as it is depleted."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, parameter, state
from ....core.model import Auxiliary, Element, Flow, Law, Rate, Sink, Source, Symbols, System
from ....core.runtime import register_model


def _energy_laws(s: Symbols) -> list[Law]:
    """Return the extraction of the reserve, its energy cost, the growth of the capacity, and the population."""
    remaining = s.reserve / s.reserve_size
    eroi = Auxiliary('eroi', 1 + (s.initial_eroi - 1) * remaining**s.decline)
    gross = Auxiliary('gross_energy', s.capacity * remaining)
    cost = Auxiliary('extraction_cost', gross.symbol / eroi.symbol)
    net = gross.symbol - cost.symbol
    return [
        eroi,
        gross,
        cost,
        Auxiliary('net_per_capita', net / s.population),
        Flow(s.reserve, Sink('extraction_work'), cost.symbol, label='energy_cost'),
        Flow(s.reserve, Sink('society'), net, label='net_energy'),
        Rate(s.capacity, s.reinvestment * net - s.capacity / s.lifetime),
        Flow(Source('births'), s.population, s.birth_rate * s.population, label='births'),
        Flow(s.population, Sink('deaths'), s.death_rate * s.population, label='deaths'),
    ]


EnergySector = Element.define(
    'EnergySector',
    states={
        'reserve': state(default=1000.0, domain=Domain.NONNEGATIVE, quantity='energy'),
        'capacity': state(default=0.1, domain=Domain.NONNEGATIVE),
        'population': state(default=1.0, domain=Domain.NONNEGATIVE, quantity='people'),
    },
    parameters={
        'reserve_size': parameter(default=1000.0, domain=Domain.POSITIVE),
        'initial_eroi': parameter(default=20.0, domain=Domain.POSITIVE),
        'decline': parameter(default=3.0, domain=Domain.POSITIVE),
        'reinvestment': parameter(default=0.1, domain=Domain.NONNEGATIVE),
        'lifetime': parameter(default=20.0, domain=Domain.POSITIVE),
        'birth_rate': parameter(default=0.03, domain=Domain.NONNEGATIVE),
        'death_rate': parameter(default=0.015, domain=Domain.NONNEGATIVE),
    },
    laws=_energy_laws,
)


@register_model(
    field='earth/climate',
    tags=('energy', 'net energy', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Mobus 2022',),
    defaults={'until': 250.0, 'dt': 1.0},
)
def energy_return(
    reserve: float = 1000.0,
    capacity: float = 0.1,
    *,
    initial_eroi: float = 20.0,
    reinvestment: float = 0.1,
    birth_rate: float = 0.03,
) -> System:
    """Return the extraction of a finite energy reserve whose energy return on investment declines as it depletes.

    The extraction capacity draws the gross energy from the reserve at a rate that falls with the remaining fraction,
    since the easy sources go first. Part of the gross energy pays for the extraction work, the gross energy divided
    by the energy return on investment (EROI), which falls from `initial_eroi` to one with the cube of the remaining
    fraction as the reserve empties. The rest is the net energy that society receives. Society reinvests a share of
    the net energy in extraction capacity, which depreciates, and the population grows exponentially with no feedback
    from the energy it receives. Energy is conserved, every unit leaving the reserve is either cost or net energy, so
    the gross energy rises exponentially and peaks, the net energy peaks before it, and the net energy per person
    peaks earlier still. The structure follows Mobus (2022, Section 9.6) and its figures of peak net energy, with
    illustrative parameters in energy units, persons, and years.

    Args:
        reserve:
            The initial energy reserve, also its full size.

        capacity:
            The initial extraction capacity, the gross energy per year from a full reserve.

        initial_eroi:
            The energy return on investment of a full reserve.

        reinvestment:
            The share of the net energy reinvested in extraction capacity.

        birth_rate:
            The relative birth rate of the population per year.

    Returns:
        system:
            The model, with the reserve `energy.reserve`, the capacity `energy.capacity`, the population
            `energy.population`, the auxiliaries `energy.eroi`, `energy.gross_energy`, `energy.extraction_cost`, and
            `energy.net_per_capita`, and the flows `energy.energy_cost`, `energy.net_energy`, `energy.births`, and
            `energy.deaths` per year, over years.

    Examples:
        >>> from skcomplex.models.earth.climate import energy_return
        >>> energy_return(initial_eroi=20.0).elements[0].initial_eroi
        20.0
    """
    sector = EnergySector(
        'energy',
        reserve=reserve,
        capacity=capacity,
        reserve_size=reserve,
        initial_eroi=initial_eroi,
        reinvestment=reinvestment,
        birth_rate=birth_rate,
    )
    return System(components=[sector])
