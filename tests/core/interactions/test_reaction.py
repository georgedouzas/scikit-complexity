"""Test the `_reaction` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.elements import CompartmentElement, Element
from skcomplex.core.interactions import ReactionInteraction
from skcomplex.core.laws import JumpLaw, NoiseLaw, RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

UNITS = 100_000
SEEDS = range(4)


class ConcentrationElement(Element):
    """Concentration `x` of a species."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x


def _epidemic(mode):
    """Return the reactions `S + I -> 2 I` and `I -> R` of 100,000 units in a mode."""
    return System(
        elements=[
            CompartmentElement('S', n=UNITS - 2_000),
            CompartmentElement('I', n=2_000),
            CompartmentElement('R'),
        ],
        interactions=[
            ReactionInteraction('infection', reactants=['S', 'I'], products=['I', 'I'], k=2.5e-6, mode=mode),
            ReactionInteraction('recovery', reactants=['I'], products=['R'], k=0.1, mode=mode),
        ],
    )


def _states(mode, random_state=None):
    """Return the amounts of `S`, `I`, and `R` over 40 time units, one row per time."""
    simulation = Simulation(_epidemic(mode), dt=1.0, random_state=random_state).check().prepare()
    trajectory = simulation.run(until=40).trajectory_
    return np.column_stack([trajectory.values(f'{name}.n') for name in ('S', 'I', 'R')])


def test_reaction_jump_means_agree_with_the_rate_solution():
    """Test that the means of the jumps stay within 2 percent of the rate solution, and both keep the units."""
    rate = _states('rate')
    jumps = [_states('jump', seed) for seed in SEEDS]
    np.testing.assert_allclose(rate.sum(axis=1), UNITS)
    for states in jumps:
        assert np.all(states.sum(axis=1) == UNITS)
        assert np.all(states == np.round(states))
    assert np.all(np.abs(np.mean(jumps, axis=0) - rate) <= 0.02 * rate.max(axis=0))


def test_reaction_langevin_conserves_the_units():
    """Test that the chemical Langevin noise of a reaction drives its states with one process, keeping the total."""
    states = _states('langevin', 0)
    np.testing.assert_allclose(states.sum(axis=1), UNITS)
    assert np.abs(states - _states('rate')).max() > 0


@pytest.mark.parametrize(
    ('mode', 'kinds'),
    [('rate', [RateLaw] * 2), ('jump', [JumpLaw]), ('langevin', [RateLaw] * 2 + [NoiseLaw] * 2)],
)
def test_reaction_runs_into_the_laws_of_its_mode(mode, kinds):
    """Test that a reaction runs into rate laws, one jump law, or rate and noise laws, by its mode."""
    a, b, k = var('a b k')
    namespace = SimpleNamespace(reactants=(SimpleNamespace(n=a),), products=(SimpleNamespace(n=b),), k=k)
    reaction = ReactionInteraction('conversion', reactants=['A'], products=['B'], mode=mode).check().prepare()
    assert [type(law) for law in reaction.run(namespace).laws_] == kinds


def test_reaction_moves_the_net_change_of_a_catalysed_reaction():
    """Test that a reactant that is also a product, a catalyst, is not changed by the reaction."""
    system = System(
        elements=[CompartmentElement('E', n=1.0), CompartmentElement('S', n=1.0), CompartmentElement('P')],
        interactions=[ReactionInteraction('catalysis', reactants=['E', 'S'], products=['E', 'P'], k=1.0)],
    )
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=2).trajectory_
    assert trajectory.values('E.n').tolist() == [1.0] * 5
    np.testing.assert_allclose(trajectory.values('S.n'), np.exp(-np.asarray(trajectory.times)), rtol=1e-4)


def test_reaction_sources_and_sinks_reach_their_balance():
    """Test that a source without reactants and a sink without products balance at `k_source / k_sink`."""
    system = System(
        elements=[ConcentrationElement('X')],
        interactions=[
            ReactionInteraction('synthesis', products=['X'], k=2.0, state='x'),
            ReactionInteraction('degradation', reactants=['X'], k=0.5, state='x'),
        ],
    )
    trajectory = Simulation(system, dt=1.0).check().prepare().run(until=40).trajectory_
    assert trajectory.values('X.x')[-1] == pytest.approx(4.0, rel=1e-4)


def test_reaction_takes_a_propensity_function():
    """Test that a reaction takes its propensity from a function of its namespace instead of mass action."""
    system = System(
        elements=[CompartmentElement('A', n=8.0), CompartmentElement('B')],
        interactions=[
            ReactionInteraction(
                'saturation',
                reactants=['A'],
                products=['B'],
                k=1.0,
                propensity=lambda s: s.k * s.reactants[0].n / (1 + s.reactants[0].n),
            ),
        ],
    )
    trajectory = Simulation(system, dt=0.001).check().prepare().run(until=0.001).trajectory_
    rate = (trajectory.values('B.n')[-1] - trajectory.values('B.n')[0]) / 0.001
    assert rate == pytest.approx(8 / 9, rel=1e-3)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'reactants': ['A'], 'mode': 'ode'}, 'has the mode'),
        ({}, 'neither reactants nor products'),
        ({'reactants': ['A'], 'state': 'not a name'}, 'not a name of a state'),
    ],
)
def test_reaction_rejects_invalid_arguments(arguments, message):
    """Test that a reaction rejects an unknown mode, no reactants and products, and a state that is not a name."""
    with pytest.raises(ValidationError, match=message):
        ReactionInteraction('reaction', **arguments).check()
