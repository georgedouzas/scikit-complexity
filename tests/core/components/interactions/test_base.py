"""Test the `_base` module of `skcomplex.core.components.interactions`."""

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import I, function, oo, sqrt, var
from sage.sets.real_set import RealSet

from skcomplex.core.base import BaseSpec
from skcomplex.core.components import RoleSpec, interactions
from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import BaseInteraction, Interaction, TrainingInteraction, WienerProcessSpec
from skcomplex.core.components.laws import (
    CallSpec,
    NoiseLaw,
    ObjectiveLaw,
    PartnerSpec,
    RateLaw,
    TotalisticSpec,
    UpdateLaw,
)
from skcomplex.core.components.laws.utils import clip
from skcomplex.core.components.spaces import (
    BaseSpace,
    ContinuousSpace,
    LatticeSpace,
    RadiusNeighborhood,
)
from skcomplex.core.engine.backends import NumbaBackend, NumpyBackend
from skcomplex.core.engine.solvers import RK4Solver, VelocityVerletSolver
from skcomplex.core.errors import ParameterShapeError, SpaceError, ValidationError
from skcomplex.core.systems import System
from skcomplex.core.types import Role
from skcomplex.systems.general.agents import (
    BirthDeathInteraction,
    ChoiceInteraction,
    EvolutionInteraction,
    ImitationInteraction,
    MovementInteraction,
)
from skcomplex.systems.general.gradients import GradientInteraction
from skcomplex.systems.general.matching import MatchingInteraction
from skcomplex.systems.general.reactions import ReactionInteraction
from skcomplex.systems.physics.common.avalanches import TopplingInteraction

RENAMED = {
    'choice': (
        ChoiceInteraction,
        lambda role: {'state': 's', 'options': (0.0, 1.0), 'utility': lambda n, option: option},
        True,
    ),
}
AGENTS = {
    'choice': (
        ChoiceInteraction,
        {'state': 's', 'options': (0.0, 1.0), 'utility': lambda agent, system, option: option},
    ),
    'imitation': (ImitationInteraction, {'state': 's', 'payoff': 's', 'probability': 0.5}),
}
AGENT_ALIASES = {
    'choice-intensity': ('choice', 'intensity', 2.0),
    'choice-activity': ('choice', 'activity', 0.5),
    'imitation-probability': ('imitation', 'probability', 0.5),
}
SYMBOLIC_ALIASES = {
    'choice-intensity': (ChoiceInteraction, 'intensity', 2.0, RENAMED['choice'][1]('target'), True),
    'choice-activity': (ChoiceInteraction, 'activity', 0.5, RENAMED['choice'][1]('target'), True),
}
GRID_ONLY = {
    'movement': (MovementInteraction('walk', target='agents', step='neighbor'), 'step'),
    'life cycle': (BirthDeathInteraction('life', target='agents', placement='neighbor'), 'placement'),
    'matching': (
        MatchingInteraction(
            'market',
            requesters='agents',
            providers='agents',
            rule=PartnerSpec('neighbors'),
        ),
        'candidates',
    ),
}
SPEC_VARIANTS = {
    'choice rule': lambda spec: ChoiceInteraction(
        'choice',
        target='members',
        state='s',
        options=(0, 1),
        utility=lambda s, o: o,
        rule=spec,
    ),
    'imitation rule': lambda spec: ImitationInteraction('imitation', target='players', rule=spec),
    'evolution selection': lambda spec: EvolutionInteraction('evolution', target='agents', selection=spec),
    'evolution replacement': lambda spec: EvolutionInteraction('evolution', target='agents', replacement=spec),
    'cascade mode': lambda spec: TopplingInteraction('cascade', target='sites', state='h', threshold=4, mode=spec),
}


class GrowthInteraction(BaseInteraction):
    """Interaction with a role and a parameter."""

    def __init__(self, label, target: Role, r: float = 1.0):
        super().__init__(label)
        self.target = target
        self.r = r

    def _prepare(self, context=None):
        """Set no laws on a namespace."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = []


def _get_fixed(interaction):
    """Return a subclass of a general interaction that declares no parameter, so it reads its attributes."""

    class FixedInteraction(interaction):
        """Interaction that declares no parameter."""

        def __init__(self, label, *, target: Role, **arguments):
            super().__init__(label, target=target, **arguments)

    return FixedInteraction


def _get_aliased(interaction):
    """Return a subclass of a general interaction that reads one parameter under the parameter `alias`."""

    class AliasInteraction(interaction):
        """Interaction that reads a renamed parameter as `alias`."""

        def __init__(
            self,
            label,
            *,
            target: Role,
            renamed: str,
            alias: float,
            **arguments,
        ):
            super().__init__(label, target=target, **arguments)
            self.renamed = renamed
            self.alias = alias

        def _get_parameter(self, name):
            """Return `alias` for the renamed parameter."""
            return 'alias' if name == self.renamed else name

    return AliasInteraction


def _run_agents(interaction):
    """Return the final values of the state `s` of four agents on a lattice driven by an agent form."""
    values = {'s': np.array([0.0, 1.0, 0.0, 1.0]), 'x': [0, 0, 1, 1], 'y': [0, 1, 0, 1]}
    agents = Element('nodes', values=values, size=4, space=LatticeSpace((2, 2)))
    system = System(elements=[agents], interactions=[interaction]).prepare()
    return system.simulate(until=5, random_state=0).get_agents('nodes', 5).values['s']


def _get_renamed(interaction):
    """Return a subclass of a general interaction that binds its target as `stock` and declares no parameter."""

    class StockInteraction(interaction):
        """Interaction that binds its target as `stock`."""

        def __init__(self, label, *, stock: Role, **arguments):
            super().__init__(label, target=stock, **arguments)
            self.stock = stock

        def _get_role(self, role):
            """Return `stock` for the target."""
            return {'target': 'stock'}.get(role, role)

    return StockInteraction


def _run_nodes(interaction):
    """Return the trajectory of the state `s` of four lattice nodes driven by an interaction in discrete time."""
    nodes = Element('nodes', values={'s': np.array([0.0, 1.0, 0.0, 1.0])}, space=LatticeSpace((2, 2)))
    system = System(elements=[nodes], interactions=[interaction]).prepare()
    return system.simulate(until=5, dt=1, random_state=0).get_values('nodes.s')


def test_interaction_stores_its_roles_and_parameters():
    """Test that an interaction stores its arguments, which scikit-learn lists."""
    assert GrowthInteraction('growth', target='rabbits', r=0.5).get_params() == {
        'label': 'growth',
        'r': 0.5,
        'target': 'rabbits',
    }


def test_interaction_prepares_into_itself():
    """Test that a checked interaction has nothing to prepare and sets its laws when prepared on a namespace."""
    growth = GrowthInteraction('growth', target='rabbits')
    assert growth.prepare() is growth
    assert growth.prepare(SimpleNamespace()).laws_ == []


@pytest.mark.parametrize('name', RENAMED)
def test_get_member_reads_a_renamed_target_in_the_symbolic_form(name):
    """Test that a subclass that binds its target as `stock` and fixes its parameters gives the plain trajectory."""
    interaction, arguments, _ = RENAMED[name]
    plain = _run_nodes(interaction(name, target='nodes', **arguments('target')))
    renamed = _run_nodes(_get_renamed(interaction)(name, stock='nodes', **arguments('stock')))
    np.testing.assert_allclose(renamed, plain)


@pytest.mark.parametrize('name', AGENTS)
def test_get_parameter_value_reads_the_attributes_of_undeclared_parameters(name):
    """Test that the agent form of a subclass that declares no parameter runs as the plain interaction."""
    interaction, arguments = AGENTS[name]
    plain = _run_agents(interaction(name, target='nodes', schedule='synchronous', **arguments))
    fixed = _run_agents(_get_fixed(interaction)(name, target='nodes', schedule='synchronous', **arguments))
    np.testing.assert_array_equal(fixed, plain)


@pytest.mark.parametrize('case', AGENT_ALIASES)
def test_get_parameter_value_reads_a_renamed_parameter_in_the_agent_form(case):
    """Test that the agent form of a subclass that reads a parameter as `alias` runs as the plain interaction."""
    name, parameter, value = AGENT_ALIASES[case]
    interaction, arguments = AGENTS[name]
    plain = _run_agents(interaction(name, target='nodes', schedule='synchronous', **{**arguments, parameter: value}))
    aliased = _get_aliased(interaction)(
        name,
        target='nodes',
        schedule='synchronous',
        renamed=parameter,
        alias=value,
        **arguments,
    )
    np.testing.assert_array_equal(_run_agents(aliased), plain)


@pytest.mark.parametrize('case', SYMBOLIC_ALIASES)
def test_get_parameter_symbol_reads_a_renamed_parameter_in_the_symbolic_form(case):
    """Test that the symbolic form of a subclass that reads a parameter as `alias` gives the plain trajectory."""
    interaction, parameter, value, arguments, _ = SYMBOLIC_ALIASES[case]
    plain = _run_nodes(interaction(case, target='nodes', **{**arguments, parameter: value}))
    aliased = _get_aliased(interaction)(case, target='nodes', renamed=parameter, alias=value, **arguments)
    np.testing.assert_allclose(_run_nodes(aliased), plain)


@pytest.mark.parametrize('case', list(GRID_ONLY))
def test_prepare_rejects_grid_variants_in_a_continuous_space(case):
    """Test that a variant of cells or nodes raises an error naming its alternative in a continuous space."""
    interaction, name = GRID_ONLY[case]
    with pytest.raises(SpaceError, match=rf'The {name} .* needs the cells of a grid .* Use '):
        interaction.prepare(ContinuousSpace(extent=(5.0, 5.0)))
    interaction.prepare(LatticeSpace((5, 5)))


@dataclass(frozen=True)
class FixedSpec(BaseSpec):
    """Variant a subclass defines, holding the value its dispatch method returns."""

    value: float = 1.0


@pytest.mark.parametrize('case', list(SPEC_VARIANTS))
def test_check_accepts_a_spec_for_a_variant_and_rejects_an_unknown_name(case):
    """Test that a variant argument takes a spec a subclass defines, and an unknown name lists the named values."""
    assert SPEC_VARIANTS[case](FixedSpec()).prepare() is not None
    match = r'Unknown .* Use one of .*, or a spec that the component handles\.'
    with pytest.raises(ValidationError, match=match) as error:
        SPEC_VARIANTS[case]('unknown').prepare()
    assert 'BaseSpec' not in str(error.value)


class FixedChoiceInteraction(ChoiceInteraction):
    """Choice that takes the value of a `FixedSpec` rule, a variant its dispatch method adds."""

    def _draw_agent(self, agent, system, intensity):
        """Return the value of a `FixedSpec` rule, or the option the parent draws by its named rule."""
        if isinstance(self.rule, FixedSpec):
            return self.rule.value
        return super()._draw_agent(agent, system, intensity)


def test_dispatch_method_of_a_subclass_runs_its_spec_variant():
    """Test that a choice subclass whose dispatch method handles a spec rule sets every agent to its value."""
    group = Element('members', values={'s': np.zeros(3)}, size=3)
    choice = FixedChoiceInteraction(
        'choice',
        target='members',
        state='s',
        options=(0, 1, 2),
        utility=print,
        rule=FixedSpec(2.0),
        schedule='synchronous',
    )
    trajectory = System(elements=[group], interactions=[choice]).prepare().simulate(until=1, random_state=0)
    assert trajectory.get_agents('members', 1).values['s'].tolist() == [2.0, 2.0, 2.0]


def _run_variant(*interactions):
    """Run three members under interactions, on a lattice of three sites."""
    group = Element('nodes', values={'s': np.zeros(3)}, size=3)
    system = System(elements=[group], interactions=list(interactions), space=LatticeSpace((3,)))
    system.prepare().simulate(until=2, random_state=0)


def _choice(schedule):
    """Return a choice with an unhandled spec rule."""
    utility = (lambda namespace, option: option) if schedule is None else (lambda agent, system, option: option)
    return ChoiceInteraction(
        'choice',
        target='nodes',
        state='s',
        options=(0, 1),
        utility=utility,
        rule=FixedSpec(),
        schedule=schedule,
    )


def _imitation(schedule, probability):
    """Return an imitation with an unhandled spec rule."""
    return ImitationInteraction(
        'imitation',
        target='nodes',
        state='s',
        options=(0, 1),
        payoff='s',
        probability=probability,
        rule=FixedSpec(),
        schedule=schedule,
    )


UNHANDLED_VARIANTS = {
    'choice agent': (lambda: _run_variant(_choice('synchronous')), '_draw_agent'),
    'choice symbolic': (lambda: _run_variant(_choice(None)), '_draw_symbolic'),
    'imitation agent': (lambda: _run_variant(_imitation('synchronous', 1.0)), '_choose'),
    'imitation rate': (lambda: _run_variant(_imitation(None, None)), '_get_jumps'),
    'imitation probability': (lambda: _run_variant(_imitation(None, 1.0)), '_get_update'),
    'evolution selection': (
        lambda: _run_variant(
            EvolutionInteraction('evolution', target='nodes', strategy='s', fitness=None, selection=FixedSpec()),
        ),
        '_select',
    ),
    'evolution replacement': (
        lambda: _run_variant(
            EvolutionInteraction('evolution', target='nodes', strategy='s', fitness=None, replacement=FixedSpec()),
        ),
        '_breed',
    ),
    'cascade': (
        lambda: _run_variant(TopplingInteraction('cascade', target='nodes', state='s', threshold=4, mode=FixedSpec())),
        '_relax',
    ),
}


@pytest.mark.parametrize('case', list(UNHANDLED_VARIANTS))
def test_dispatch_raises_for_a_spec_variant_without_an_override(case):
    """Test that a spec variant no subclass handles raises an error naming the method to override."""
    run, method = UNHANDLED_VARIANTS[case]
    with pytest.raises(ValidationError, match=f'Override `{method}`'):
        run()


def _interaction_classes() -> list[type[BaseInteraction]]:
    """Return the public interaction classes of the core."""
    named = [getattr(interactions, name) for name in interactions.__all__]
    return [item for item in named if isinstance(item, type) and issubclass(item, BaseInteraction)]


def test_get_requirements_reads_the_names_of_a_tuple_of_states():
    """Test that an argument typed `tuple[str, ...]` names the states its role reads, and a mapping names none."""
    gradient = GradientInteraction('flow', target='a', states=('x', 'y'), potential=lambda s: 0.0)
    assert gradient.get_requirements().roles['target'].states == ('x', 'y')
    reaction = ReactionInteraction('r', target='a', reactants={'S': 1, ('b', 'n'): 1}, products={'P': 1})
    assert reaction.get_requirements().roles['target'].states == ()


def test_get_requirements_leaves_unknown_the_laws_a_subclass_sets():
    """Test that a subclass that sets its laws in `_prepare` without listing them lists none."""
    needs = PullInteraction('pull').get_requirements()
    assert needs.laws == ()
    assert needs.time == ()


def test_get_requirements_accepts_any_space_when_a_law_does():
    """Test that laws that include one running in any space state no space."""
    assert ReactionInteraction('decay', target='a', reactants={'x': 1}).get_requirements().space == ()


def test_every_core_interaction_declares_its_roles():
    """Test that every role of every interaction of the core names the kinds of element it accepts."""
    found = _interaction_classes()
    assert {kind.__name__ for kind in found} == {name for name in interactions.__all__ if name.endswith('Interaction')}
    for kind in found:
        for name, spec in kind._get_specs().items():
            if isinstance(spec, RoleSpec) and not spec.noise:
                assert spec.elements, f'{kind.__name__}.{name}'


def test_every_core_interaction_with_laws_lists_them():
    """Test that every interaction of the core lists its laws.

    The bases a subclass writes laws for and the training, whose solver reads its settings, state none.
    """
    bases = {BaseInteraction, Interaction}
    for kind in set(_interaction_classes()) - bases - {TrainingInteraction}:
        assert kind.law_classes, kind.__name__


@pytest.mark.parametrize(
    ('interaction', 'laws', 'randomness'),
    [
        (ReactionInteraction('decay', target='a', reactants={'x': 1}), {'rate'}, 'deterministic'),
        (ReactionInteraction('decay', target='a', reactants={'x': 1}, mode='jump'), {'jump'}, 'stochastic'),
        (
            ReactionInteraction('decay', target='a', reactants={'x': 1}, mode='langevin'),
            {'rate', 'noise'},
            'stochastic',
        ),
    ],
)
def test_interaction_tags_follow_its_arguments(interaction, laws, randomness):
    """Test that the law kinds and the randomness of an interaction follow the arguments that choose its laws."""
    tags = interaction.get_tags()
    assert (tags.laws, tags.randomness) == (laws, randomness)


def test_interaction_tags_give_the_times_and_structure_of_the_chosen_laws():
    """Test that an interaction gives the times its chosen laws run in, and how they change the structure."""
    assert ReactionInteraction('decay', target='a', reactants={'x': 1}).get_tags().time == {'continuous'}
    assert TopplingInteraction('cascade', target='a', state='h', threshold=4).get_tags().time == {'discrete'}


def test_interaction_that_sets_unlisted_laws_has_unknown_law_kinds():
    """Test that a subclass that sets its laws without listing them leaves its law kinds and randomness unknown."""
    tags = PullInteraction('pull').get_tags()
    assert (tags.laws, tags.randomness) == (None, None)
    assert Interaction('free').get_tags().laws is None


class PullInteraction(Interaction):
    """Interaction that learns a constant pull on its target when it is prepared."""

    def __init__(
        self,
        label: str,
        *,
        target: Role = 'ball',
        g: float = 9.81,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.g = g

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the pull as the law of the interaction."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x.diff(t, 2), -context.g)]


def test_prepare_sets_laws_from_a_namespace():
    """Preparing an interaction on a namespace of states as functions of time sets its laws."""
    t, g = var('t g')
    x = function('x')(t)
    pull = PullInteraction('pull').prepare(SimpleNamespace(target=SimpleNamespace(x=x), g=g, t=t))
    assert [(law.target, law.value) for law in pull.laws_] == [(x.diff(t, 2), -g)]


def test_prepare_keeps_parameters_symbolic_in_a_system():
    """The laws a system learns from an interaction read its parameters as symbols."""
    system = System(elements=[Element('ball', values={'x': 10.0})], interactions=[PullInteraction('pull')]).prepare()
    assert system.equations_["ball.x'"] == '-pull.g'


def test_prepare_leaves_the_user_interaction_unchanged():
    """The engine prepares a clone of an interaction for every binding, so the object the user gave has no laws."""
    pull = PullInteraction('pull', target='bodies')
    bodies = Element('bodies', values={'x': [1.0, 2.0]}, size=2)
    System(elements=[bodies], interactions=[pull]).prepare().simulate(until=1.0, dt=0.5)
    assert not hasattr(pull, 'laws_')


class RuleInteraction(Interaction):
    """Interaction whose law is an update of the group of its target with a given value."""

    def __init__(self, label, *, target: Role, rule=None, bound=True):
        super().__init__(label)
        self.target = target
        self.rule = rule
        self.bound = bound

    def _prepare(self, context):
        """Set the update of the bound group, or of a namespace no role binds."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            target = context.target if self.bound else SimpleNamespace()
            self.laws_ = [UpdateLaw(target, self.rule, phase='move')]


def _shift(agent, system):
    """Add one to the value of an agent."""
    agent.x += 1


def test_update_of_a_group_with_a_function_is_an_agent_rule():
    """Test that an update law whose target is a bound group runs its function for every member of the role."""
    walkers = SimpleNamespace(x=var('x'))
    laws = RuleInteraction('shift', target='walkers', rule=_shift)._get_laws_of(SimpleNamespace(target=walkers))
    assert [(law.function, law.members, law.phase) for law in laws] == [(_shift, 'target', 'move')]


@pytest.mark.parametrize('rule', [3.0, lambda agent: None, lambda agent, system, extra: None])
def test_rule_that_is_no_function_of_an_agent_and_the_system_raises(rule):
    """Test that the value of an update law of a group must be a function of one agent and the system."""
    interaction = RuleInteraction('shift', target='walkers', rule=rule)
    with pytest.raises(ValidationError, match='A rule is a function of one agent and the system'):
        interaction._get_laws_of(SimpleNamespace(target=SimpleNamespace(x=var('x'))))


class ParityInteraction(Interaction):
    """Interaction whose law sets every cell to the parity of its neighbourhood."""

    def __init__(self, label, *, target: Role = 'cells', phase=None):
        super().__init__(label)
        self.target = target
        self.phase = phase

    def _prepare(self, context):
        """Set the parity of the cells as a local rule."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            parity = TotalisticSpec(lambda total: total % 2)
            neighborhood = RadiusNeighborhood(1, center=True)
            self.laws_ = [UpdateLaw(context.target.alive, parity, neighborhood=neighborhood, phase=self.phase)]


def test_update_with_a_neighborhood_is_a_local_rule():
    """Test that an update law with a neighbourhood runs as a local rule of the states of the cells."""
    alive = var('alive')
    laws = ParityInteraction('parity')._get_laws_of(SimpleNamespace(target=SimpleNamespace(alive=alive)))
    assert [(law.targets, law.label) for law in laws] == [(alive, '')]
    assert isinstance(laws[0].rule, TotalisticSpec)


def test_local_rule_with_a_setting_of_one_member_raises():
    """Test that a local rule rejects a phase, which only an update of one member takes."""
    interaction = ParityInteraction('parity', phase='move')
    with pytest.raises(ValidationError, match='neighbourhood and phase'):
        interaction._get_laws_of(SimpleNamespace(target=SimpleNamespace(alive=var('alive'))))


def test_rule_on_a_namespace_no_role_binds_raises():
    """Test that an update law targets the group of a role, not a namespace of its own."""
    interaction = RuleInteraction('shift', target='walkers', rule=_shift, bound=False)
    with pytest.raises(ValidationError, match='which no role binds'):
        interaction._get_laws_of(SimpleNamespace(target=SimpleNamespace(x=var('x'))))


class DecayInteraction(Interaction):
    """Decay `x' = -k x` at a positive rate."""

    def __init__(self, label, *, target: Role = 'p', k: Annotated[float, RealSet.open(0, oo)] = 1.0):
        super().__init__(label)
        self.target = target
        self.k = k

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x.diff(t), -context.k * x)]


def test_allowed_values_of_an_argument_come_from_annotated():
    """Test that `Annotated[float, RealSet.open(0, oo)]` gives the rate its domain, which a system checks."""
    assert DecayInteraction('decay')._get_declared()['k'].spec.domain == RealSet.open(0, oo)
    system = System(elements=[Element('p', values={'x': 1.0})], interactions=[DecayInteraction('decay', k=-1.0)])
    with pytest.raises(ValidationError, match=r'decay\.k'):
        system.prepare()


class HalvingInteraction(Interaction):
    """Update `x <- f(x)` through a function of the state."""

    def __init__(self, label, *, target: Role = 'p'):
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_ = [UpdateLaw(x, CallSpec(_halve, (x,)))]


def _halve(t, x):
    """Return half of a value."""
    return x / 2


def test_call_of_a_function_reads_the_states_of_its_arguments():
    """Test that the arguments of a `CallSpec` in a law read the states, as the rest of the law does."""
    system = System(elements=[Element('p', values={'x': 8.0})], interactions=[HalvingInteraction('halve')]).prepare()
    assert system.simulate(until=2, dt=1).get_values('p.x').tolist() == [8.0, 4.0, 2.0]


class TrapInteraction(Interaction):
    """Trap `x' = -k x` with one stiffness per axis, written before a label setting and a tuple setting."""

    def __init__(self, label, *, target: Role = 'bead', k=None, name='trap', axes=(0, 1)):
        super().__init__(label)
        self.target = target
        self.k = [1.0, 2.0] if k is None else k
        self.name = name
        self.axes = axes

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x = context.target.x
            self.laws_ = [RateLaw(x, -context.k.pairwise_product(x))]


def test_list_argument_is_a_vector_parameter_that_a_schedule_drives():
    """Test that a list of numbers is a vector parameter of its length, which a schedule changes during a run."""
    declared = TrapInteraction('trap')._get_declared()
    assert list(declared) == ['k']
    assert declared['k'].spec.vector
    assert declared['k'].spec.n_dim == 2
    bead = Element('bead', values={'x': [1.0, 1.0]})
    system = System(elements=[bead], interactions=[TrapInteraction('trap')]).prepare()
    assert system.parameters_ == ('trap.k[0]', 'trap.k[1]')
    final = system.simulate(until=1.0, dt=0.5).get_values('bead.x')[-1]
    assert final == pytest.approx(np.exp([-1.0, -2.0]), rel=1e-4)
    driven = system.simulate(until=1.0, dt=0.5, schedule={'trap.k': lambda t: [3.0, 0.0]})
    assert driven.get_values('bead.x')[-1] == pytest.approx(np.exp([-3.0, 0.0]), rel=1e-4)


def test_matrix_argument_is_a_matrix_parameter():
    """Test that a square matrix of numbers is a parameter that laws read as a matrix."""
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    declared = TrapInteraction('trap', k=rotation)._get_declared()['k']
    assert declared.spec.tensor == 'full'
    assert np.asarray(declared.value).tolist() == [0.0, -1.0, 1.0, 0.0]


class SpringInteraction(Interaction):
    """Spring `x'' = -k (x - a)` whose stiffness comes before its rest position."""

    def __init__(self, label, *, target: Role = 'p', k=1.0, a=0.0):
        super().__init__(label)
        self.target = target
        self.k = k
        self.a = a

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            x, t = context.target.x, context.t
            self.laws_ = [RateLaw(x.diff(t, 2), -context.k * (x - context.a))]


def test_parameters_of_an_interaction_keep_the_order_of_its_arguments():
    """Test that the parameters come in the order of the arguments, not the alphabetical one."""
    system = System(elements=[Element('p', values={'x': 1.0})], interactions=[SpringInteraction('spring')]).prepare()
    assert system.parameters_ == ('spring.k', 'spring.a')


def langevin(noise):
    """Return a damped particle whose velocity `x'` is driven by the given noise law."""

    def laws(s):
        x, t = s.target.x, s.t
        return [RateLaw(x.diff(t, 2), -x.diff(t)), noise(x, t)]

    interaction = Interaction('kick', roles={'target': 'p'}, laws=laws)
    return System(elements=[Element('p', values={'x': 0.0})], interactions=[interaction]).prepare()


def test_noise_on_a_derivative_is_written_in_sage():
    """Test that `NoiseLaw(x.diff(t, 2), g)` drives the velocity `x'` as a noise on a velocity state `v` does."""
    sage = langevin(lambda x, t: NoiseLaw(x.diff(t, 2), 0.5)).simulate(until=1.0, dt=0.1, random_state=0)

    def laws(s):
        x, v, t = s.target.x, s.target.v, s.t
        return [RateLaw(x.diff(t), v), RateLaw(v.diff(t), -v), NoiseLaw(v.diff(t), 0.5)]

    kick = Interaction('kick', roles={'target': 'p'}, laws=laws)
    first = System(elements=[Element('p', values={'x': 0.0, 'v': 0.0})], interactions=[kick]).prepare()
    explicit = first.simulate(until=1.0, dt=0.1, random_state=0)
    assert np.ptp(sage.get_values("p.x'")) > 0
    assert sage.get_values("p.x'") == pytest.approx(explicit.get_values('p.v'))


def test_noise_of_a_lower_order_names_the_derivative_to_write():
    """Test that a noise on `x` of a state of order 2 asks for `x.diff(t, 2)`."""
    with pytest.raises(ValidationError, match=r'x\.diff\(t, 2\)'):
        langevin(lambda x, t: NoiseLaw(x, 0.5))


class _SharedNoiseInteraction(Interaction):
    """Noise of one value driven by the Wiener processes it is given."""

    def __init__(self, label, *, target: Role, process=None):
        super().__init__(label)
        self.target = target
        self.process = process

    def _prepare(self, context=None):
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [NoiseLaw(context.target.x, 1.0, process=context.process.dW)]


def test_an_argument_holding_wiener_processes_binds_them():
    """Two interactions given the same processes, with no annotation, share their increments."""
    market = WienerProcessSpec('market')
    system = System(
        elements=[Element('a', values={'x': 0.0}), Element('b', values={'x': 0.0})],
        interactions=[
            _SharedNoiseInteraction('first', target='a', process=market),
            _SharedNoiseInteraction('second', target='b', process=market),
        ],
    )
    trajectory = system.prepare().simulate(until=1.0, dt=0.1, random_state=0)
    assert np.ptp(trajectory.get_values('a.x')) > 0
    np.testing.assert_allclose(trajectory.get_values('a.x'), trajectory.get_values('b.x'))


def _get_delayed_decay(shift):
    """Return the run of a decay that reads its value a shift of time before."""
    decay = Interaction(
        'decay',
        roles={'target': 'n'},
        laws=lambda s: [
            RateLaw(s.target.x.diff(s.t), -s.target.x(t=s.t - shift(s.target))),
            RateLaw(s.target.tau.diff(s.t), -0.1 * s.target.tau),
        ],
    )
    neuron = Element('n', values={'x': 1.0, 'tau': 0.5})
    system = System(elements=[neuron], interactions=[decay], solver=RK4Solver(dt=0.01)).prepare()
    return system, system.simulate(until=2.0).get_values('n.x')


def test_a_delay_that_reads_the_state_is_bounded_by_its_clip():
    """Test that a delay of the state clipped from below compiles, and runs as the fixed delay of its bound."""
    system, clipped = _get_delayed_decay(lambda target: clip(target.tau * target.x, low=0.6))
    assert 'where(n.tau*n.x < 0.6, 0.6, n.tau*n.x)' in system.equations_['n.x']
    _, fixed = _get_delayed_decay(lambda target: 0.6)
    np.testing.assert_allclose(clipped, fixed, rtol=1e-12)


def _get_rotated(target: Callable[[SimpleNamespace], object]) -> object:
    """Return the last value of a complex point whose derivative of an order is `i z`."""
    rotate = Interaction(
        'rotate',
        roles={'point': 'point'},
        laws=lambda s: [RateLaw(target(s), I * s.point.z)],
    )
    system = System(elements=[Element('point', values={'z': 1.0 + 0.0j})], interactions=[rotate]).prepare()
    return system.simulate(until=1.0).get_values('point.z')[-1]


def test_rate_of_the_derivative_of_a_complex_state_is_its_rate():
    """Test that `RateLaw(z.diff(t), f)` of a complex state turns it as `RateLaw(z, f)` does, by `exp(i t)`."""
    by_derivative = _get_rotated(lambda s: s.point.z.diff(s.t))
    assert by_derivative == pytest.approx(_get_rotated(lambda s: s.point.z))
    assert by_derivative == pytest.approx(np.exp(1j), rel=1e-5)


def test_second_derivative_of_a_complex_state_gives_it_a_second_order():
    """Test that `z'' = i z` from rest at one gives `cosh(sqrt(i) t)` at the end."""
    assert _get_rotated(lambda s: s.point.z.diff(s.t, 2)) == pytest.approx(np.cosh(np.sqrt(1j)), rel=1e-5)


class Contraction(Interaction):
    """Decay `x' = -a[0, 0, 0] x` of one element at a rate read from a tensor."""

    def __init__(self, label: str, *, target: Role = 'ball', a: object = None) -> None:
        super().__init__(label)
        self.target = target
        self.a = a

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the decay at the first entry of the tensor."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            self.laws_ = [RateLaw(context.target.x, -context.a[0][0][0] * context.target.x)]


def test_array_argument_no_group_shapes_is_an_issue_of_the_shape_of_a_parameter():
    """Test that a tensor argument read by the laws of one element is named as an issue, not an attribute error."""
    contraction = Contraction('contraction', a=np.ones((2, 2, 2)))
    system = System(elements=[Element('ball', values={'x': 1.0})], interactions=[contraction])
    with pytest.raises(
        ParameterShapeError,
        match=r'contraction\.a is an array of shape \(2, 2, 2\).*\[parameter-shape\]',
    ):
        system.prepare()


def _distance(s):
    """Return the separation and the distance of the two members of a pair."""
    separation = s.first.x - s.second.x
    return separation, sqrt(separation.dot_product(separation))


def _lennard_jones_energy(s):
    """Return the Lennard-Jones energy `4 * (r**-12 - r**-6)` of a pair."""
    _, r = _distance(s)
    return [ObjectiveLaw(4 * (r**-12 - r**-6))]


def _lennard_jones_forces(s, mass=1.0):
    """Return the Lennard-Jones accelerations of a pair of a mass, derived by hand."""
    separation, r = _distance(s)
    force = 24 * (2 * r**-12 - r**-6) / r**2 * separation / mass
    return [RateLaw(s.first.x.diff(s.t, 2), force), RateLaw(s.second.x.diff(s.t, 2), -force)]


def _lennard_jones_gas(laws, backend, masses=None):
    """Return the positions of a run of a Lennard-Jones gas of 16 atoms on a jittered lattice."""
    rng = np.random.default_rng(0)
    grid = np.stack(np.meshgrid(np.arange(4), np.arange(4)), axis=-1).reshape(-1, 2) * 1.2
    space = ContinuousSpace(n_dim=2)
    values = {'x': grid + rng.normal(0, 0.05, (16, 2)), "x'": rng.normal(0, 0.5, (16, 2))} | (masses or {})
    gas = Element('gas', values=values, size=16, space=space)
    pairs = Interaction('lj', roles={'first': 'gas', 'second': 'gas'}, within=RadiusNeighborhood(2.5), laws=laws)
    solver = VelocityVerletSolver(dt=0.005)
    system = System(elements=[gas], interactions=[pairs], space=space, solver=solver, backend=backend())
    return system.prepare().simulate(until=0.5, dt=0.05).get_values('gas.x')


@pytest.mark.parametrize('backend', [NumpyBackend, NumbaBackend])
def test_pair_energy_gives_the_trajectory_of_its_forces(backend):
    """Test that a Lennard-Jones energy of pairs moves a gas as the Lennard-Jones forces derived by hand do."""
    energy = _lennard_jones_gas(_lennard_jones_energy, backend)
    forces = _lennard_jones_gas(_lennard_jones_forces, backend)
    assert np.abs(energy[-1] - energy[0]).max() > 0.1
    assert energy == pytest.approx(forces, abs=1e-9)


def test_pair_energy_divides_the_forces_by_the_mass():
    """Test that atoms of mass 4 under a Lennard-Jones energy move as the forces divided by 4."""
    energy = _lennard_jones_gas(_lennard_jones_energy, NumpyBackend, {'m': 4.0})
    forces = _lennard_jones_gas(lambda s: _lennard_jones_forces(s, 4.0), NumpyBackend)
    assert energy == pytest.approx(forces, abs=1e-9)


def test_pair_energy_with_targets_is_rejected():
    """Test that a pair energy naming targets raises an error, since it acts on the positions of both members."""
    space = ContinuousSpace(n_dim=1)
    gas = Element('gas', values={'x': np.array([[0.0], [1.0]])}, size=2, space=space)
    pairs = Interaction(
        'soft',
        roles={'first': 'gas', 'second': 'gas'},
        within=RadiusNeighborhood(2.0),
        laws=lambda s: [ObjectiveLaw(_distance(s)[1] ** 2, s.first.x[0])],
    )
    with pytest.raises(ValidationError, match='names targets or controls'):
        System(elements=[gas], interactions=[pairs], space=space).prepare()
