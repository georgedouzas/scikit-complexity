"""Net energy models: a finite fossil reserve whose energy return on investment falls as it is depleted."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.laws import BaseLaw, RateLaw, Sink, Source, TransferLaw
from ....core.specs import ObservableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model


def _eroi(s: SimpleNamespace) -> Expression:
    """Return the energy return on investment, which falls with the remaining fraction of the reserve."""
    return 1 + (s.initial_eroi - 1) * (s.reserve / s.reserve_size) ** s.decline


def _gross_energy(s: SimpleNamespace) -> Expression:
    """Return the gross energy extracted per year."""
    return s.capacity * s.reserve / s.reserve_size


def _extraction_cost(s: SimpleNamespace) -> Expression:
    """Return the energy the extraction work costs per year."""
    return _gross_energy(s) / _eroi(s)


def _net_per_capita(s: SimpleNamespace) -> Expression:
    """Return the net energy per person."""
    return (_gross_energy(s) - _extraction_cost(s)) / s.population


def _energy_laws(s: SimpleNamespace) -> list[BaseLaw]:
    """Return the extraction of the reserve, its energy cost, the growth of the capacity, and the population."""
    cost = _extraction_cost(s)
    net = _gross_energy(s) - cost
    return [
        TransferLaw(s.reserve, Sink('extraction_work'), cost, label='energy_cost'),
        TransferLaw(s.reserve, Sink('society'), net, label='net_energy'),
        RateLaw(s.capacity, s.reinvestment * net - s.capacity / s.lifetime),
        TransferLaw(Source('births'), s.population, s.birth_rate * s.population, label='births'),
        TransferLaw(s.population, Sink('deaths'), s.death_rate * s.population, label='deaths'),
    ]


class EnergySectorElement(Element):
    """Energy sector element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        reserve: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='energy')] = 1000.0,
        capacity: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        population: Annotated[object, StateSpec(domain=RealSet.closed_open(0, oo), quantity='people')] = 1.0,
        reserve_size: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 1000.0,
        initial_eroi: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        decline: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        reinvestment: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        lifetime: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 20.0,
        birth_rate: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.03,
        death_rate: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.015,
    ) -> None:
        super().__init__(label, visual=visual)
        self.reserve = reserve
        self.capacity = capacity
        self.population = population
        self.reserve_size = reserve_size
        self.initial_eroi = initial_eroi
        self.decline = decline
        self.reinvestment = reinvestment
        self.lifetime = lifetime
        self.birth_rate = birth_rate
        self.death_rate = death_rate

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the laws of the energy sector element."""
        return _energy_laws(s)


@register_model(
    field='earth/climate',
    tags=('energy', 'net energy', 'stocks and flows', 'system dynamics', 'ode'),
    references=('Mobus 2022',),
    defaults={'until': 250.0, 'dt': 1.0},
    observables=lambda system: [
        ObservableSpec('eroi', _eroi, element='energy'),
        ObservableSpec('gross_energy', _gross_energy, element='energy'),
        ObservableSpec('extraction_cost', _extraction_cost, element='energy'),
        ObservableSpec('net_per_capita', _net_per_capita, element='energy'),
    ],
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
            `energy.population`, and the flows `energy.energy_cost`, `energy.net_energy`, `energy.births`, and
            `energy.deaths` per year, over years. The observables of the model are `energy.eroi`,
            `energy.gross_energy`, `energy.extraction_cost`, and `energy.net_per_capita`.

    Examples:
        >>> from skcomplex.models.earth.climate import energy_return
        >>> energy_return(initial_eroi=20.0).elements[0].initial_eroi
        20.0
    """
    sector = EnergySectorElement(
        'energy',
        reserve=reserve,
        capacity=capacity,
        reserve_size=reserve,
        initial_eroi=initial_eroi,
        reinvestment=reinvestment,
        birth_rate=birth_rate,
    )
    return System(elements=[sector])
