"""Test the `_reaction` module of `skcomplex.core.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.elements import CompartmentElement, Element
from skcomplex.core.interactions import JumpInteraction, ReactionInteraction
from skcomplex.core.laws import JumpLaw, NoiseLaw, RateLaw
from skcomplex.core.system import System
from skcomplex.engine.runtime import Simulation
from skcomplex.errors import ValidationError
from skcomplex.specs import StateSpec, VisualSpec

UNITS = 100_000
SEEDS = range(4)


class ConcentrationElement(Element):
    """Concentrations `x` and `y` of two species."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        y: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y


def _epidemic(mode):
    """Return the reactions `S + I -> 2 I` and `I -> R` of 100,000 units in a mode."""
    return System(
        elements=[
            CompartmentElement('S', n=UNITS - 2_000),
            CompartmentElement('I', n=2_000),
            CompartmentElement('R'),
        ],
        interactions=[
            ReactionInteraction(
                'infection',
                reactants={('S', 'n'): 1, ('I', 'n'): 1},
                products={('I', 'n'): 2},
                k=2.5e-6,
                mode=mode,
            ),
            ReactionInteraction('recovery', reactants={('I', 'n'): 1}, products={('R', 'n'): 1}, k=0.1, mode=mode),
        ],
    )


def _states(mode, random_state=None):
    """Return the amounts of `S`, `I`, and `R` over 40 time units, one row per time."""
    simulation = Simulation(_epidemic(mode), dt=1.0, random_state=random_state).check().prepare()
    trajectory = simulation.run(until=40).trajectory_
    return np.column_stack([trajectory.values(f'{name}.n') for name in ('S', 'I', 'R')])


def test_reaction_is_a_jump():
    """Test that a reaction is a jump whose clock is its propensity."""
    assert issubclass(ReactionInteraction, JumpInteraction)


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
    namespace = SimpleNamespace(
        reactants=(SimpleNamespace(n=a),),
        products=(SimpleNamespace(n=b),),
        modifiers=(),
        k=k,
    )
    reaction = ReactionInteraction('conversion', reactants={('A', 'n'): 1}, products={('B', 'n'): 1}, mode=mode)
    assert [type(law) for law in reaction.check().prepare().run(namespace).laws_] == kinds


def test_reaction_reads_modifiers_without_changing_them():
    """Test that a modifier enters the mass-action propensity, and the reaction does not change it."""
    system = System(
        elements=[CompartmentElement('E', n=1.0), CompartmentElement('S', n=1.0), CompartmentElement('P')],
        interactions=[
            ReactionInteraction(
                'catalysis',
                reactants={('S', 'n'): 1},
                products={('P', 'n'): 1},
                modifiers={('E', 'n'): 1},
                k=1.0,
            ),
        ],
    )
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=2).trajectory_
    assert trajectory.values('E.n').tolist() == [1.0] * 5
    np.testing.assert_allclose(trajectory.values('S.n'), np.exp(-np.asarray(trajectory.times)), rtol=1e-4)


def test_reaction_moves_states_of_one_element_with_real_coefficients():
    """Test that `2 x -> 1.5 y` on two states of one element moves the units by its real stoichiometry."""
    system = System(
        elements=[ConcentrationElement('cell', x=1.0)],
        interactions=[
            ReactionInteraction('dimerization', reactants={('cell', 'x'): 2}, products={('cell', 'y'): 1.5}, k=0.5),
        ],
    )
    trajectory = Simulation(system, dt=0.5).check().prepare().run(until=4).trajectory_
    x, y = trajectory.values('cell.x'), trajectory.values('cell.y')
    times = np.asarray(trajectory.times)
    np.testing.assert_allclose(x, 1 / (1 + times), rtol=1e-4)
    np.testing.assert_allclose(y, 0.75 * (1 - x), rtol=1e-4)


def test_reaction_sources_and_sinks_reach_their_balance():
    """Test that a source without reactants and a sink without products balance at `k_source / k_sink`."""
    system = System(
        elements=[ConcentrationElement('X')],
        interactions=[
            ReactionInteraction('synthesis', products={('X', 'x'): 1}, k=2.0),
            ReactionInteraction('degradation', reactants={('X', 'x'): 1}, k=0.5),
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
                reactants={('A', 'n'): 1},
                products={('B', 'n'): 1},
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
        ({'reactants': {('A', 'n'): 1}, 'mode': 'ode'}, 'has the mode'),
        ({}, 'neither reactants nor products'),
        ({'reactants': ['A']}, 'Map pairs of a path and a state name'),
        ({'reactants': {('A', 'not a name'): 1}}, 'Map pairs of a path and a state name'),
        ({'reactants': {('A', 'n'): -1}}, 'positive coefficients'),
    ],
)
def test_reaction_rejects_invalid_arguments(arguments, message):
    """Test that a reaction rejects an unknown mode, no species, and entries or coefficients it cannot take."""
    with pytest.raises(ValidationError, match=message):
        ReactionInteraction('reaction', **arguments).check()
