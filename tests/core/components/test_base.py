"""Test the `base` module of `skcomplex.core.components`."""

import inspect
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated

import pytest
from sage.all__sagemath_symbolics import var
from sklearn.base import clone

from skcomplex.core.base import BaseConfig
from skcomplex.core.components import RoleSpec
from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import BaseElement, Element
from skcomplex.core.components.interactions import BaseInteraction, Interaction
from skcomplex.core.components.laws import BaseLaw, RateLaw
from skcomplex.core.components.spaces import BaseNeighborhood, BaseSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.systems import BaseSystem, System
from skcomplex.core.types import Role


class DoublingComponent(BaseComponent):
    """Component that doubles its factor after counting its preparations."""

    def __init__(self, label, factor=2):
        super().__init__(label)
        self.factor = factor

    def _prepare(self, context=None):
        """Check the factor, then count the preparation and keep the context."""
        super()._prepare(context)
        if not isinstance(self.factor, int | float):
            error_msg = f'The factor {self.factor!r} is not a number.'
            raise ValidationError(error_msg)
        self.prepared_ = getattr(self, 'prepared_', 0) + 1
        self.context_ = context
        self.doubled_ = 2 * self.factor


class SpecifiedComponent(BaseComponent):
    """Component that declares two roles, a value, and one plain argument."""

    def __init__(self, label, x: float = 0.0, target: Role = 'a', source: Role | None = None, plain=None):
        super().__init__(label)
        self.x = x
        self.target = target
        self.source = source
        self.plain = plain

    def _prepare(self, context=None):
        """Check the label and the declared specs, and prepare nothing else."""
        super()._prepare(context)


class ExtendedComponent(SpecifiedComponent):
    """Component that takes the role of its base again without annotating it, and adds a role first."""

    def __init__(self, label, other: Role = 'b', target='a'):
        super().__init__(label, target=target)
        self.other = other


def test_prepare_derives_the_attributes():
    """Test that preparing derives the attributes of the component from a context and returns the component."""
    component = DoublingComponent('doubling', factor=3)
    assert component.prepare('context') is component
    assert (component.prepared_, component.context_, component.doubled_) == (1, 'context', 6)
    assert component.prepare().prepared_ == 2


def test_prepare_checks_the_label_and_the_arguments():
    """Test that preparing rejects a label that is not a string and an argument the component cannot take."""
    with pytest.raises(ValidationError, match='label of DoublingComponent must be a string'):
        DoublingComponent(1).prepare()
    with pytest.raises(ValidationError, match='not a number'):
        DoublingComponent('doubling', factor='two').prepare()


def test_get_specs_returns_the_annotated_arguments_in_constructor_order():
    """Test that the specs of a component are the annotated arguments of its constructor, in order, and no others."""
    specs = SpecifiedComponent._get_specs()
    assert list(specs) == ['target', 'source']
    assert specs['target'] == RoleSpec()
    assert specs['source'] == RoleSpec(optional=True)


def test_get_specs_keeps_the_specs_of_the_bases():
    """Test that a subclass keeps the spec of an argument of its base that it takes again without annotating it."""
    specs = ExtendedComponent._get_specs()
    assert list(specs) == ['other', 'target']
    assert specs['target'] == RoleSpec()


def test_constructor_only_stores_its_arguments():
    """Test that a component stores its arguments as given, so scikit-learn lists and clones them."""
    component = SpecifiedComponent('a', x=[1.0], plain='plain')
    assert component.get_params() == {'label': 'a', 'plain': 'plain', 'source': None, 'target': 'a', 'x': [1.0]}
    cloned = clone(component)
    assert cloned.get_params() == component.get_params()
    assert cloned is not component


def test_components_derive_from_the_component_base():
    """Test that every component of a system is a component, and no component is an engine class."""
    components = (BaseElement, BaseInteraction, BaseLaw, System, BaseSpace, BaseNeighborhood)
    assert all(issubclass(component, BaseComponent) for component in components)
    assert not any(issubclass(component, BaseConfig) for component in components)


def _get_public_methods(klass):
    """Return the names of the public methods of a class."""
    return {name for name, value in inspect.getmembers(klass) if not name.startswith('_') and callable(value)}


@pytest.mark.parametrize(
    ('base', 'verbs'),
    [
        (BaseComponent, set()),
        (BaseElement, {'get_states'}),
        (BaseInteraction, {'get_requirements'}),
        (BaseLaw, {'get_requirements'}),
        (BaseSpace, set()),
        (BaseNeighborhood, set()),
        (BaseSystem, {'simulate'}),
    ],
)
def test_bases_list_their_methods(base, verbs):
    """Test that a component base has the parameter methods, tags, and `prepare`, and its own verb if any."""
    common = {'get_params', 'set_params', 'get_metadata_routing', 'get_tags', 'prepare'}
    assert _get_public_methods(base) == common | verbs


def test_cloning_shares_the_symbolic_values():
    """Test that a clone of a component is a new component with the same parameters and shared Sage values."""
    x = var('x')
    law = RateLaw(x, -2 * x, label='decay')
    cloned = clone(law)
    assert cloned is not law
    assert cloned.get_params() == law.get_params()
    assert cloned.value is law.value


class VoteInteraction(Interaction):
    """Vote of a group whose role accepts only Boolean values."""

    def __init__(self, label: str, *, voters: Annotated[Role, 'boolean'], other: Role = 'others') -> None:
        super().__init__(label)
        self.voters = voters
        self.other = other


def test_get_specs_reads_the_domains_written_after_a_role():
    """Test that `Annotated[Role, 'boolean']` is a role that accepts Boolean values, and a plain `Role` any value."""
    specs = VoteInteraction._get_specs()
    assert specs['voters'] == RoleSpec(domains=('boolean',))
    assert specs['other'] == RoleSpec()


class ReactionInteraction(Interaction):
    """Interaction whose roles bind one element, one element or none, and several elements."""

    def __init__(
        self,
        label: str,
        *,
        reactants: tuple[Role, ...] = (),
        amounts: Mapping[tuple[Role, str] | Role, float] | None = None,
        catalyst: Role | None = None,
        states: tuple[str, ...] = ('x',),
        k: float = 1.0,
    ) -> None:
        super().__init__(label)
        self.reactants = reactants
        self.amounts = amounts
        self.catalyst = catalyst
        self.states = states
        self.k = k


def test_get_specs_reads_roles_from_type_hints():
    """Test that `tuple[Role, ...]` and a mapping keyed by roles bind several elements and `Role | None` is optional."""
    specs = ReactionInteraction._get_specs()
    assert specs['reactants'] == RoleSpec(variadic=True)
    assert specs['amounts'] == RoleSpec(variadic=True, optional=True)
    assert specs['catalyst'] == RoleSpec(optional=True)
    assert 'states' not in specs
    assert list(ReactionInteraction('reaction')._get_declared()) == ['k']


class FeedInteraction(Interaction):
    """Decay of a stock fed by an optional source."""

    def __init__(self, label: str, *, target: Role, source: Role | None = None, k: float = 1.0) -> None:
        super().__init__(label)
        self.target = target
        self.source = source
        self.k = k

    def _prepare(self, context=None):
        """Set the decay, fed by the source when one is bound."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            feed = 0 if self.source is None else context.source.x
            self.laws_ = [RateLaw(context.target.x.diff(context.t), -context.k * context.target.x + feed)]


@pytest.mark.parametrize(('source', 'expected'), [(None, '-feed.k*a.x'), ('b', '-feed.k*a.x + b.x')])
def test_a_role_typed_role_or_none_is_optional(source, expected):
    """Test that `Role | None` is an optional role, which binds an element or nothing."""
    assert FeedInteraction._get_specs()['source'] == RoleSpec(optional=True)
    elements = [Element('a', values={'x': 1.0}), Element('b', values={'x': 2.0})]
    system = System(elements=elements, interactions=[FeedInteraction('feed', target='a', source=source)]).prepare()
    assert system.equations_['a.x'] == expected
