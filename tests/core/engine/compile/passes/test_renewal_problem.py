"""Test the compilation of renewal laws."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import pytest

from skcomplex.core.components.elements import Element, GroupElement
from skcomplex.core.components.laws import JumpLaw, RateLaw, UpdateLaw
from skcomplex.core.components.laws.utils import draw, mean, normal
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.enums import FormKind
from skcomplex.core.errors import ValidationError
from skcomplex.core.specs import StateSpec, VisualSpec


class CounterElement(Element):
    """Counter element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n

    def _get_laws(self, s):
        """Return the laws of the counter element."""
        return [JumpLaw(waiting=draw('wait', 'expon'), reset={s.n: s.n + 1})]


class DecayElement(Element):
    """Decay element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        y: Annotated[object, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.y = y

    def _get_laws(self, s):
        """Return the laws of the decay element."""
        return [RateLaw(s.y, -s.y)]


class RenewingElement(Element):
    """Element with one renewal law of a waiting time, a function of the namespace of its symbols."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
        waiting: Callable[[SimpleNamespace], object] = lambda s: draw('wait', 'expon'),
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.waiting = waiting

    def _get_laws(self, s):
        """Return the renewal law of the renewing element."""
        return [JumpLaw(waiting=self.waiting(s))]


class MeanResetElement(Element):
    """Element whose renewal law resets its state to the mean over its group."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x

    def _get_laws(self, s):
        """Return the renewal law of the mean reset element."""
        return [JumpLaw(waiting=draw('wait', 'expon'), reset={s.x: mean(s.x)})]


def test_compiler_gives_renewal_laws_alone_the_renewal_form():
    """Test that a system of renewal laws alone compiles to the renewal form with a hidden count per member."""
    system = CompositeSystem(elements=[GroupElement('counters', element=CounterElement, size=3)])
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.form == FormKind.RENEWAL
    assert problem.states.names == ['counters.n', 'counters.counters~renewals']
    assert problem.renewals[0].label == 'counters'
    assert problem.renewals[0].members == 3


def test_compiler_partitions_renewals_beside_an_ode():
    """Test that a renewal subsystem and an ODE subsystem in sibling subsystems compile to a partitioned problem."""
    system = CompositeSystem(
        systems=[
            CompositeSystem('renewing', elements=[CounterElement('c')]),
            CompositeSystem('decaying', elements=[DecayElement('d')]),
        ],
    )
    problem = PassCompiler().prepare(system).run().problem_
    assert problem.form == FormKind.COUPLED
    assert [partition.problem.form for partition in problem.partitions] == [FormKind.RENEWAL, FormKind.ODE]


def test_compiler_keeps_the_form_of_rate_laws_with_renewals():
    """Test that renewal laws beside rate laws of the same component keep the ordinary differential form."""

    class WalkerElement(Element):
        """Walker element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
            v: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.v = v

        def _get_laws(self, s):
            """Return the laws of the walker element."""
            return [RateLaw(s.x, s.v), JumpLaw(waiting=draw('wait', 'expon'), reset={s.v: -s.v})]

    walker = WalkerElement
    assert PassCompiler().prepare(CompositeSystem(elements=[walker('w')])).run().problem_.form == FormKind.ODE


def test_compiler_rejects_an_unknown_distribution():
    """Test that a draw from an unknown distribution names the law, the distribution, and the closest names."""
    typo = RenewingElement('r', waiting=lambda s: draw('wait', 'paretto', b=1.5))
    with pytest.raises(ValidationError, match=r"RenewalLaw of r draws 'wait' from the distribution 'paretto'.*pareto"):
        PassCompiler().prepare(CompositeSystem(elements=[typo])).run()


def test_compiler_rejects_unknown_keywords():
    """Test that a draw with a keyword its distribution does not take lists the keywords it takes."""
    wrong = RenewingElement('r', waiting=lambda s: draw('wait', 'pareto', alpha=1.5))
    with pytest.raises(ValidationError, match=r'pareto takes b, loc, scale'):
        PassCompiler().prepare(CompositeSystem(elements=[wrong])).run()


@pytest.mark.parametrize(
    'waiting',
    [
        lambda s: normal('wait', 1, 1),
        lambda s: draw('wait', 'norm', loc=1),
        lambda s: draw('wait', 'levy_stable', alpha=1.5, beta=0),
    ],
)
def test_compiler_rejects_waiting_times_that_can_be_negative(waiting):
    """Test that a waiting time from a distribution with negative values asks for nonnegative support."""
    with pytest.raises(ValidationError, match=r'renewal law r of r draws its waiting time.*nonnegative support'):
        PassCompiler().prepare(CompositeSystem(elements=[RenewingElement('r', waiting=waiting)])).run()


def test_compiler_rejects_reductions_in_renewal_laws():
    """Test that a renewal law that reads a reduction over its group raises an error."""
    with pytest.raises(ValidationError, match=r'reads a reduction over a group'):
        PassCompiler().prepare(CompositeSystem(elements=[GroupElement('g', element=MeanResetElement, size=3)])).run()


def test_compiler_accepts_draws_in_update_laws_from_scipy():
    """Test that a SciPy draw in an update law takes one value per member after the uniform and normal draws."""

    class JumpingElement(Element):
        """Jumping element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the jumping element."""
            return [UpdateLaw(s.x, s.x + draw('jump', 'lomax', c=2.0) + normal('noise'))]

    jumping = JumpingElement
    problem = (
        PassCompiler().prepare(CompositeSystem(elements=[GroupElement('g', element=jumping, size=4)])).run().problem_
    )
    assert problem.draws is not None
    assert problem.draws.size == 8
    assert problem.draws.sampled[0].rows == 4
