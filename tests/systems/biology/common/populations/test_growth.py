"""Test the `_growth` module of `skcomplex.systems.biology.common.populations`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import Element
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec
from skcomplex.systems.biology.common.kinetics import CompartmentElement
from skcomplex.systems.biology.common.populations import GrowthInteraction

N0 = 2.0
K = 100.0


class BiomassElement(Element):
    """Biomass `x` of a population."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


def _run(until=10.0, dt=0.05, **arguments):
    """Return the times and the population of one compartment that grows by the given arguments."""
    system = CompositeSystem(
        elements=[CompartmentElement('population', x=N0)],
        interactions=[GrowthInteraction('growth', target='population', **arguments)],
    )
    trajectory = simulate(system, dt=dt, until=until)
    return np.asarray(trajectory.times), np.asarray(trajectory.get_values('population.x'))


def _bertalanffy(t, r, p):
    """Return the closed solution of von Bertalanffy's growth towards `K`, linear in `n ** (1 - p)`."""
    limit = K ** (1 - p)
    return (limit - (limit - N0 ** (1 - p)) * np.exp(-(1 - p) * r * K ** (p - 1) * t)) ** (1 / (1 - p))


@pytest.mark.parametrize(
    ('arguments', 'solution'),
    [
        ({'form': 'exponential', 'r': 0.3}, lambda t: N0 * np.exp(0.3 * t)),
        ({'form': 'exponential', 'r': -0.3}, lambda t: N0 * np.exp(-0.3 * t)),
        ({'form': 'logistic', 'r': 1.0, 'K': K}, lambda t: K / (1 + (K / N0 - 1) * np.exp(-t))),
        (
            {'form': 'logistic', 'r': 1.0, 'K': K, 'p': 2.0},
            lambda t: K / (1 + ((K / N0) ** 2 - 1) * np.exp(-2 * t)) ** 0.5,
        ),
        ({'form': 'gompertz', 'r': 0.5, 'K': K}, lambda t: K * np.exp(np.log(N0 / K) * np.exp(-0.5 * t))),
        ({'form': 'power', 'r': 0.5, 'p': 0.5}, lambda t: (N0**0.5 + 0.5 * 0.5 * t) ** 2),
        ({'form': 'bertalanffy', 'r': 2.0, 'K': K, 'p': 2 / 3}, lambda t: _bertalanffy(t, 2.0, 2 / 3)),
    ],
)
def test_growth_follows_the_closed_solution_of_its_form(arguments, solution):
    """Test that every form of growth follows its closed solution."""
    times, population = _run(**arguments)
    np.testing.assert_allclose(population, solution(times), rtol=1e-4)


def test_growth_logistic_inflects_at_half_the_capacity():
    """Test that logistic growth from 1 approaches the capacity 100, and grows fastest at 50."""
    system = CompositeSystem(
        elements=[CompartmentElement('population', x=1.0)],
        interactions=[GrowthInteraction('growth', target='population', r=1.0, K=100.0, form='logistic')],
    )
    trajectory = simulate(system, dt=0.01, until=20)
    population = np.asarray(trajectory.get_values('population.x'))
    assert population[-1] == pytest.approx(100.0, rel=1e-6)
    inflection = population[np.argmax(np.diff(population))]
    assert inflection == pytest.approx(50.0, rel=1e-2)


def test_growth_harvest_lowers_the_logistic_equilibrium():
    """Test that a proportional harvest `h` lowers the logistic equilibrium to `K (1 - h / r)`."""
    _, population = _run(until=60.0, form='logistic', r=1.0, K=K, h=0.25)
    assert population[-1] == pytest.approx(K * 0.75, rel=1e-6)


def test_growth_delay_reads_the_history_before_one_delay():
    """Test that a delayed logistic grows exponentially at `r (1 - n0 / K)` within its first delay."""
    times, population = _run(until=2.0, dt=0.01, form='logistic', r=1.0, K=K, tau=2.0)
    np.testing.assert_allclose(population, N0 * np.exp((1 - N0 / K) * times), rtol=1e-4)


def test_growth_long_delay_sustains_oscillations():
    """Test that a delayed logistic with `r tau` above `pi / 2` oscillates around the capacity."""
    times, population = _run(until=200.0, dt=0.05, form='logistic', r=1.0, K=K, tau=2.0)
    late = population[times > 150]
    assert late.min() < 0.5 * K < 1.5 * K < late.max()


def test_growth_reads_the_named_state():
    """Test that growth changes the state its argument names."""
    system = CompositeSystem(
        elements=[BiomassElement('cells', x=1.0)],
        interactions=[GrowthInteraction('growth', target='cells', r=0.5, state='x')],
    )
    trajectory = simulate(system, dt=1.0, until=2)
    assert trajectory.get_values('cells.x')[-1] == pytest.approx(np.exp(1.0), rel=1e-4)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'form': 'richards'}, 'has the form'),
        ({'state': '1n'}, 'not a name of a state'),
        ({'form': 'power', 'tau': 1.0}, 'has no limiting factor'),
        ({'form': 'bertalanffy', 'p': 1.0}, 'exponent 1'),
    ],
)
def test_growth_rejects_invalid_arguments(arguments, message):
    """Test that growth rejects an unknown form, a bad state, a delay without limit, and a Bertalanffy exponent 1."""
    with pytest.raises(ValidationError, match=message):
        GrowthInteraction('growth', target='population', **arguments).prepare()
