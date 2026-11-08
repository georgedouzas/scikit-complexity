"""Test the `_reaction` module of `skcomplex.core.components.interactions`."""

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import oo, var
from sage.sets.real_set import RealSet

from skcomplex.core.components.elements import CompartmentElement, Element, FieldElement, GroupElement
from skcomplex.core.components.interactions import JumpInteraction, ReactionInteraction
from skcomplex.core.components.laws import JumpLaw, NoiseLaw, RateLaw
from skcomplex.core.components.spaces import DiscreteTime, LatticeSpace
from skcomplex.core.components.system import System
from skcomplex.core.engine.runtime import simulate
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec

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
    trajectory = simulate(_epidemic(mode), until=40, dt=1.0, random_state=random_state)
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
    assert [type(law) for law in reaction.prepare().run(namespace).laws_] == kinds


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
    trajectory = simulate(system, until=2, dt=0.5)
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
    trajectory = simulate(system, until=4, dt=0.5)
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
    trajectory = simulate(system, until=40, dt=1.0)
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
    trajectory = simulate(system, until=0.001, dt=0.001)
    rate = (trajectory.values('B.n')[-1] - trajectory.values('B.n')[0]) / 0.001
    assert rate == pytest.approx(8 / 9, rel=1e-3)


@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'reactants': {('A', 'n'): 1}, 'mode': 'ode'}, 'has the mode'),
        ({}, 'neither reactants nor products'),
        ({'reactants': ['A']}, 'Map pairs of a path and a state name'),
        ({'reactants': {('A', 'not a name'): 1}}, 'Map pairs of a path and a state name'),
        ({'reactants': {('A', 'n'): -1}}, 'positive numbers or functions'),
        ({'reactants': {('A', 'n'): 1}, 'form': 'mesa'}, 'has the form'),
        ({'reactants': {('A', 'n'): 1}, 'form': 'agent'}, 'agent form but no target'),
        ({'target': 'cells', 'reactants': {'n': 1}, 'form': 'agent', 'mode': 'langevin'}, 'Use rate or jump'),
        ({'target': 'cells', 'reactants': {('A', 'n'): 1}, 'form': 'agent'}, 'state names alone'),
        ({'reactants': {'n': 1}}, 'state names of the target'),
    ],
)
def test_reaction_rejects_invalid_arguments(arguments, message):
    """Test that a reaction rejects an unknown mode, no species, and entries or coefficients it cannot take."""
    with pytest.raises(ValidationError, match=message):
        ReactionInteraction('reaction', **arguments).prepare()


class ConversionInteraction(ReactionInteraction):
    """Conversion `A -> B` at the rate `c A` between the states `n` of the elements of its roles `source` and `sink`."""

    def __init__(
        self,
        label: str,
        *,
        source: Annotated[str, RoleSpec()],
        sink: Annotated[str, RoleSpec()],
        c: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.source = source
        self.sink = sink
        self.c = c

    def _get_reactants(self):
        """Return the state `n` of the source."""
        return {(self.source, 'n'): 1}

    def _get_products(self):
        """Return the state `n` of the sink."""
        return {(self.sink, 'n'): 1}

    def _get_rate(self, namespace):
        """Return the mass action at the rate constant `c`."""
        return self._get_mass_action(namespace, namespace.c)


class SiteReactionInteraction(ReactionInteraction):
    """Reaction that binds its field as `sites` and declares no parameter."""

    def __init__(self, label: str, *, sites: Annotated[str, RoleSpec()], **arguments: object) -> None:
        super().__init__(label, target=sites, **arguments)
        self.sites = sites

    def _get_role(self, role: str) -> str:
        """Return `sites` for the target."""
        return 'sites' if role == 'target' else role


def test_reaction_runs_at_every_site_of_a_field():
    """Test that a reaction on state names of a field target reacts at every site with the states of the site."""
    values = {'x': np.array([1.0, 2.0, 4.0])}
    cells = FieldElement('cells', element=ConcentrationElement, space=LatticeSpace((3,)), values=values)
    dimerization = ReactionInteraction('dimerization', target='cells', reactants={'x': 2}, products={'y': 1.5}, k=0.5)
    trajectory = simulate(System(elements=[cells], interactions=[dimerization]), until=4, dt=0.5)
    x, y = trajectory.lattice('cells.x')[-1], trajectory.lattice('cells.y')[-1]
    np.testing.assert_allclose(x, values['x'] / (1 + values['x'] * 4), rtol=1e-4)
    np.testing.assert_allclose(y, 0.75 * (values['x'] - x), rtol=1e-4)


def test_reaction_jumps_in_every_member_of_a_group():
    """Test that jumps on a group target move whole units in every member on its own and keep each member's total."""
    values = {'x': np.array([50.0, 100.0])}
    group = GroupElement('cells', element=ConcentrationElement, size=2, values=values)
    conversion = ReactionInteraction('conversion', target='cells', reactants={'x': 1}, products={'y': 1}, mode='jump')
    states = simulate(System(elements=[group], interactions=[conversion]), until=1, dt=0.5, random_state=0).states
    totals = states[:, :2] + states[:, 2:]
    assert np.all(totals == values['x'])
    assert np.all(states == np.round(states))
    assert np.all(states[-1, :2] < values['x'])


def test_reaction_subclass_reads_species_under_its_own_roles():
    """Test that a subclass binding its species under roles of its own names follows its mass action."""
    system = System(
        elements=[CompartmentElement('A', n=8.0), CompartmentElement('B')],
        interactions=[ConversionInteraction('conversion', source='A', sink='B', c=0.5)],
    )
    trajectory = simulate(system, until=2, dt=0.5)
    times = np.asarray(trajectory.times)
    np.testing.assert_allclose(trajectory.values('A.n'), 8 * np.exp(-0.5 * times), rtol=1e-4)
    np.testing.assert_allclose(trajectory.values('A.n') + trajectory.values('B.n'), 8.0)


class ScaledElement(Element):
    """Concentrations `x` and `y` with the time scales `epsilon` and `delta` that divide their changes."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 1.0,
        y: Annotated[float, StateSpec()] = 0.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.25,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.epsilon = epsilon
        self.delta = delta


@pytest.mark.parametrize(('epsilon', 'delta'), [(0.5, 0.25), (0.25, 1.0)])
def test_reaction_scales_changes_by_parametric_coefficients(epsilon, delta):
    """Test that coefficients that are functions of parameters scale the change of every state by their values."""
    conversion = ReactionInteraction(
        'conversion',
        target='box',
        reactants={'x': lambda s: 1 / s.target.epsilon},
        products={'y': lambda s: 1 / s.target.delta},
        propensity=lambda s: s.target.x,
    )
    system = System(elements=[ScaledElement('box', epsilon=epsilon, delta=delta)], interactions=[conversion])
    trajectory = simulate(system, until=1, dt=0.01)
    x = np.exp(-np.asarray(trajectory.times) / epsilon)
    np.testing.assert_allclose(trajectory.values('box.x'), x, rtol=1e-4)
    np.testing.assert_allclose(trajectory.values('box.y'), epsilon / delta * (1 - x), rtol=1e-4, atol=1e-8)


def _agent_decay(mode, size=2000):
    """Return cells of three units each that decay on their own in the agent form."""
    cells = GroupElement('cells', element=CompartmentElement, size=size, values={'n': np.full(size, 3.0)})
    decay = ReactionInteraction('decay', target='cells', reactants={'n': 1}, k=0.1, form='agent', mode=mode)
    return System(elements=[cells], interactions=[decay], time=DiscreteTime())


def test_reaction_agent_form_steps_every_member_by_its_rate():
    """Test that the agent form in the mode `rate` adds the changes times the propensity of every member per step."""
    trajectory = simulate(_agent_decay('rate', size=3), until=3)
    np.testing.assert_allclose(trajectory.agents('cells', 3).values['n'], 3 * 0.9**3)


def test_reaction_agent_form_jumps_whole_units_at_the_step_probability():
    """Test that the agent form in the mode `jump` moves one unit with the probability `1 - exp(-propensity)`."""
    trajectory = simulate(_agent_decay('jump'), until=1, random_state=0)
    counts = trajectory.agents('cells', 1).values['n']
    assert set(counts.tolist()) <= {2.0, 3.0}
    assert np.mean(counts == 2.0) == pytest.approx(1 - np.exp(-0.3), abs=0.03)


@pytest.mark.parametrize('schedule', ['sequential', 'random-sequential', 'member'])
def test_reaction_agent_schedules_react_in_every_member_once(schedule):
    """Test that every agent schedule reacts once in every member per step, by its rate or at the step probability."""
    rate = _agent_decay('rate', size=3)
    rate.interactions[0].set_params(schedule=schedule)
    trajectory = simulate(rate, until=3)
    np.testing.assert_allclose(trajectory.agents('cells', 3).values['n'], 3 * 0.9**3)
    jump = _agent_decay('jump')
    jump.interactions[0].set_params(schedule=schedule)
    counts = simulate(jump, until=1, random_state=0).agents('cells', 1).values['n']
    assert np.mean(counts == 2.0) == pytest.approx(1 - np.exp(-0.3), abs=0.03)


def test_reaction_symbolic_form_refuses_an_agent_schedule():
    """Test that a symbolic reaction with a schedule other than synchronous raises an error."""
    with pytest.raises(ValidationError, match='runs agent rules'):
        ReactionInteraction('decay', target='cells', reactants={'n': 1}, schedule='member').prepare()


@pytest.mark.parametrize('mode', ['rate', 'jump'])
def test_get_laws_reads_a_renamed_target(mode):
    """Test that a reaction that binds its field as `sites` gives the trajectory of the plain reaction."""
    values = {'x': np.array([10.0, 20.0, 40.0])}
    cells = FieldElement('cells', element=ConcentrationElement, space=LatticeSpace((3,)), values=values)
    arguments = {'reactants': {'x': 2}, 'products': {'y': 1}, 'k': 0.05, 'mode': mode}
    states = []
    for dimerization in (
        ReactionInteraction('dimerization', target='cells', **arguments),
        SiteReactionInteraction('dimerization', sites='cells', **arguments),
    ):
        states.append(
            simulate(System(elements=[cells], interactions=[dimerization]), until=2, dt=0.5, random_state=0).states,
        )
    np.testing.assert_allclose(states[1], states[0])
    assert not np.allclose(states[0][-1], states[0][0])
