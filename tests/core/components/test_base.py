"""Test the `base` module of `skcomplex.core.components`."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Annotated

import pytest
from sklearn.base import clone

from skcomplex.core.components.base import BaseComponent
from skcomplex.core.components.elements import BaseElement, Element
from skcomplex.core.components.interactions import BaseInteraction, Interaction
from skcomplex.core.components.laws import BaseLaw, RateLaw
from skcomplex.core.components.spaces import BaseNeighborhood, BaseSpace
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import RoleSpec
from skcomplex.core.systems import System
from skcomplex.core.types import Role


class DoublingComponent(BaseComponent):
    """Component that doubles its input after counting its preparations."""

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

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Set `doubled_`, the input times the factor."""
        self.doubled_ = self.factor * inputs


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

    def _get_laws(self, namespace):
        """Return no laws."""
        return []

    def _run(self, inputs):
        """Run nothing."""


class ExtendedComponent(SpecifiedComponent):
    """Component that takes the role of its base again without annotating it, and adds a role first."""

    def __init__(self, label, other: Role = 'b', target='a'):
        super().__init__(label, target=target)
        self.other = other


def test_prepare_and_run():
    """Test that preparing derives the attributes of the component from a context, and running acts."""
    component = DoublingComponent('doubling').prepare('context')
    assert (component.prepared_, component.context_) == (1, 'context')
    assert component.run(3).doubled_ == 6


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


def test_element_runs_into_its_laws():
    """Test that running an element sets its laws for the namespace, as interactions and systems do."""

    class StillElement(Element):
        def _get_laws(self, namespace):
            return []

    assert StillElement('a').prepare().run(SimpleNamespace()).laws_ == []


def test_components_derive_from_the_component_base():
    """Test that every component of a system is a component, and a component cannot be made without its hooks."""
    components = (BaseElement, BaseInteraction, BaseLaw, System, BaseSpace, BaseNeighborhood)
    assert all(issubclass(component, BaseComponent) for component in components)
    with pytest.raises(TypeError, match='abstract'):
        BaseComponent('component')


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
