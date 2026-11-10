"""Test the `_transfer` module of `skcomplex.core.components.laws`."""

from typing import Annotated

import numpy as np
import pytest
from sage.all__sagemath_symbolics import var

from skcomplex.core.analysis import loops, symbolic
from skcomplex.core.components.elements import Element
from skcomplex.core.components.laws import BoundaryLaw, Sink, Source, TransferLaw, UpdateLaw
from skcomplex.core.components.laws.utils import shift_member, where
from skcomplex.core.components.spaces import DiscreteTime
from skcomplex.core.components.systems import CompositeSystem
from skcomplex.core.engine.compile._pass_compiler import PassCompiler
from skcomplex.core.engine.runtime import ConservationObserver, simulate
from skcomplex.core.errors import ConservationError, LabelError, ValidationError
from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec


class TanksElement(Element):
    """Two stocks with a transfer between them."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[float, StateSpec()] = 3.0,
        b: Annotated[float, StateSpec()] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b

    def _get_laws(self, s):
        """Return the transfer."""
        return [TransferLaw(s.a, s.b, s.a / 4, label='flow')]


def _tub_laws(s):
    """Return a faucet of 5 and a drain proportional to the level."""
    return [
        TransferLaw(Source(), s.water, 5.0, label='faucet'),
        TransferLaw(s.water, Sink(), s.k * s.water, label='drain'),
    ]


def _population_laws(s):
    """Return births and deaths proportional to the population."""
    return [
        TransferLaw(Source(), s.n, s.b * s.n, label='births'),
        TransferLaw(s.n, Sink(), s.d * s.n, label='deaths'),
    ]


def _clipped_laws(s):
    """Return an inflow and a boundary law that clips the stock, which destroys the quantity."""
    return [TransferLaw(Source(), s.water, 1.0, label='faucet'), BoundaryLaw(s.water, where(s.water > 2, 2, s.water))]


def _ledger_laws(s):
    """Return a per-step payment between two holders and an untagged counter updated every step."""
    return [
        TransferLaw(s.a, s.b, 0.1 * s.a, label='pay'),
        TransferLaw(Source(), s.a, 1.0, label='lend'),
        UpdateLaw(s.c, s.c + 1),
    ]


class TubElement(Element):
    """Tub element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        water: Annotated[object, StateSpec(quantity='water')] = 0.0,
        k: Annotated[object, ParameterSpec()] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.water = water
        self.k = k

    def _get_laws(self, s):
        """Return the laws of the tub element."""
        return _tub_laws(s)


class PopulationElement(Element):
    """Population element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        n: Annotated[object, StateSpec(quantity='people')] = 10.0,
        b: Annotated[object, ParameterSpec()] = 0.03,
        d: Annotated[object, ParameterSpec()] = 0.01,
    ) -> None:
        super().__init__(label, visual=visual)
        self.n = n
        self.b = b
        self.d = d

    def _get_laws(self, s):
        """Return the laws of the population element."""
        return _population_laws(s)


class ClippedElement(Element):
    """Clipped element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        water: Annotated[object, StateSpec(quantity='water')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.water = water

    def _get_laws(self, s):
        """Return the laws of the clipped element."""
        return _clipped_laws(s)


class LedgerElement(Element):
    """Ledger element."""

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[object, StateSpec(quantity='money')] = 10.0,
        b: Annotated[object, StateSpec(quantity='money')] = 0.0,
        c: Annotated[object, StateSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.c = c

    def _get_laws(self, s):
        """Return the laws of the ledger element."""
        return _ledger_laws(s)


@pytest.mark.parametrize('time', [None, DiscreteTime()])
def test_transfer_conserves_across_time_classes(time):
    """Test that a transfer keeps the total of two stocks in continuous and in discrete time."""
    trajectory = simulate(CompositeSystem(elements=[TanksElement('t')], time=time), until=5, dt=1.0)
    np.testing.assert_allclose(trajectory.get_values('t.a') + trajectory.get_values('t.b'), 4.0, atol=1e-12)
    assert trajectory.get_values('t.a')[-1] < 3.0


def test_transfer_from_the_environment_takes_a_source():
    """Test that a transfer from `Source()` draws the quantity from the environment, and `None` is rejected."""

    class FilledElement(Element):
        """Filled element."""

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
            """Return the laws of the filled element."""
            return [TransferLaw(Source(), s.n, 1.0, label='inflow')]

    trajectory = simulate(CompositeSystem(elements=[FilledElement('f')]), until=3, dt=1.0)
    assert trajectory.get_values('f.n')[-1] == pytest.approx(3.0)
    with pytest.raises(ValidationError, match='or `Source\\(\\)`, but got None'):
        TransferLaw(None, var('n'), 1.0).prepare()


def test_bathtub_approaches_the_balance_of_its_flows_and_records_them():
    """Test that the stock approaches inflow / k = 10, with both flows recorded and the inflow accumulated."""
    trajectory = simulate(CompositeSystem(elements=[TubElement('tub')]), until=30, dt=1)
    assert trajectory.get_values('tub.water')[-1] == pytest.approx(10.0, rel=1e-3)
    assert trajectory.get_values('tub.faucet')[-1] == pytest.approx(5.0)
    assert trajectory.get_values('tub.drain')[-1] == pytest.approx(5.0, rel=1e-3)
    assert trajectory.get_values('__boundary.water')[-1] == pytest.approx(
        trajectory.get_values('tub.water')[-1],
        rel=1e-6,
    )


def test_flow_between_quantities_is_rejected():
    """Test that a flow between holders of different quantities raises a conservation error."""

    class MixedElement(Element):
        """Mixed element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec(quantity='mass')] = 1.0,
            y: Annotated[object, StateSpec(quantity='energy')] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.y = y

        def _get_laws(self, s):
            """Return the laws of the mixed element."""
            return [TransferLaw(s.x, s.y, s.x)]

    mixed = MixedElement
    with pytest.raises(ConservationError, match='different quantities'):
        PassCompiler().prepare(CompositeSystem(elements=[mixed('m')])).run()


def test_ends_of_a_transfer_print_and_compare_by_their_fields():
    """Test that a source and a sink print as their fields, compare by them, and appear so in an error."""
    assert (repr(Source()), repr(Sink('drain'))) == ("Source(label='environment')", "Sink(label='drain')")
    assert Source() == Source('environment')
    assert Sink('drain') != Sink()
    with pytest.raises(ValidationError, match=r"or `Source\(\)`, but got Sink\(label='environment'\)\."):
        TransferLaw(Sink(), Source(), 1.0).prepare()


def test_flow_holders_and_labels_are_checked():
    """Test that flows reject parameters as holders, environment to environment, and duplicate labels."""
    with pytest.raises(ValidationError, match='both of its holders'):
        TransferLaw(Source(), Sink(), 1.0).prepare()

    class TwiceElement(Element):
        """Twice element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x

        def _get_laws(self, s):
            """Return the laws of the twice element."""
            return [TransferLaw(Source(), s.x, 1.0, label='f'), TransferLaw(s.x, Sink(), 1.0, label='f')]

    twice = TwiceElement
    with pytest.raises(LabelError, match="'f'"):
        PassCompiler().prepare(CompositeSystem(elements=[twice('twice')])).run()

    class ReadsElement(Element):
        """Reads element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            x: Annotated[object, StateSpec()] = 1.0,
            k: Annotated[object, ParameterSpec()] = 1.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.x = x
            self.k = k

        def _get_laws(self, s):
            """Return the laws of the reads element."""
            return [TransferLaw(s.k, s.x, 1)]

    reads = ReadsElement
    with pytest.raises(ValidationError, match='not a state'):
        PassCompiler().prepare(CompositeSystem(elements=[reads('reads')])).run()


def test_flow_takes_one_holder_read_at_a_member_offset():
    """Test that one holder of a flow may be a state read at a member offset, but not both of them."""
    x = var('x')
    assert TransferLaw(x, shift_member(x, 1), 1.0).prepare().targets_ == (x, shift_member(x, 1))
    with pytest.raises(ValidationError, match='Both holders'):
        TransferLaw(shift_member(x, -1), shift_member(x, 1), 1.0).prepare()


def test_flow_owner_reports_the_flow_under_another_part():
    """Test that an owner reports a flow under its path, and that two flows reported as one path are rejected."""

    class ReportedElement(TanksElement):
        """Tanks whose flow is reported under the other tanks."""

        def _get_laws(self, s):
            """Return the transfer reported under the other tanks."""
            return [TransferLaw(s.a, s.b, s.a / 4, label='flow', owner='other')]

    system = CompositeSystem(elements=[ReportedElement('tanks'), TanksElement('spare')])
    trajectory = simulate(system, until=1, dt=1)
    assert 'other.flow' in trajectory.names
    assert 'tanks.flow' not in trajectory.names
    with pytest.raises(LabelError, match="'flow'"):
        PassCompiler().prepare(CompositeSystem(elements=[ReportedElement('tanks'), TanksElement('other')])).run()
    with pytest.raises(ValidationError, match='owner'):
        TransferLaw(var('a'), var('b'), 1.0, owner=3).prepare()


def test_discrete_flows_are_per_step_and_join_the_updates():
    """Test that in a discrete map a flow moves its amount once per step and money changes only by lending."""
    trajectory = simulate(CompositeSystem(elements=[LedgerElement('ledger')]), until=5, dt=1)
    assert trajectory.get_values('ledger.a').tolist() == pytest.approx([10.0] * 6)
    assert trajectory.get_values('ledger.b').tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert trajectory.get_values('__boundary.money')[-1] == pytest.approx(5.0)
    assert trajectory.get_values('ledger.c')[-1] == pytest.approx(5.0)


def test_jump_flows_move_their_size_at_random_times():
    """Test that a flow with a rate becomes a jump that conserves the total of its holders."""

    class JumpingElement(Element):
        """Jumping element."""

        def __init__(
            self,
            label: str,
            visual: VisualSpec | None = None,
            *,
            a: Annotated[object, StateSpec(quantity='coins')] = 50.0,
            b: Annotated[object, StateSpec(quantity='coins')] = 0.0,
        ) -> None:
            super().__init__(label, visual=visual)
            self.a = a
            self.b = b

        def _get_laws(self, s):
            """Return the laws of the jumping element."""
            return [TransferLaw(s.a, s.b, 1, rate=0.1 * s.a, label='move')]

    jumping = JumpingElement
    trajectory = simulate(CompositeSystem(elements=[jumping('j')]), until=10, dt=1, random_state=0)
    assert np.all(trajectory.get_values('j.a') + trajectory.get_values('j.b') == 50.0)
    assert trajectory.get_values('j.b')[-1] > 0


def test_population_has_one_reinforcing_and_one_balancing_loop():
    """Test that births close a reinforcing loop and deaths a balancing loop, and births dominate."""
    system = CompositeSystem(elements=[PopulationElement('pop')])
    analysis = loops(system)
    found = dict(zip(analysis.loops, analysis.polarities, strict=True))
    assert found == {('pop.n', 'pop.deaths'): 'balancing', ('pop.n', 'pop.births'): 'reinforcing'}
    trajectory = simulate(system, until=5, dt=1)
    dominance = loops(trajectory, kind='dominance')
    assert dominance.gains[-1] == pytest.approx([0.03, -0.01])
    assert dominance.polarities[int(dominance.dominant[-1])] == 'reinforcing'


def test_conservation_returns_the_boundary_flows():
    """Test that the net inflow of people is births minus deaths."""
    boundary = symbolic(CompositeSystem(elements=[PopulationElement('pop')]), kind='conservation').inflows
    assert str(boundary['people']) == 'b_pop*n_pop - d_pop*n_pop'


def test_monitor_detects_drift_beyond_the_flows():
    """Test that clipping a stock with a boundary law destroys water, which the monitor reports."""
    with pytest.raises(ConservationError, match="'water' drifted"):
        simulate(CompositeSystem(elements=[ClippedElement('c')]), until=5, dt=1, observers=[ConservationObserver()])
