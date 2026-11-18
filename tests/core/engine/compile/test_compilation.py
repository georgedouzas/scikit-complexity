"""Test the `_compilation` module of `skcomplex.core.engine.compile`."""

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import Interaction
from skcomplex.core.components.laws import ConstraintLaw
from skcomplex.core.components.spaces import ContinuousSpace
from skcomplex.core.specs import Role
from skcomplex.core.systems import System


class BalanceInteraction(Interaction):
    """Two conditions on two quantities, `x + y = 3` and `x - y = 1`."""

    def __init__(self, label: str, *, target: Role = 'pair') -> None:
        super().__init__(label)
        self.target = target

    def _prepare(self, context=None):
        """Set the two conditions as the laws."""
        super()._prepare(context)
        x, y = context.target.x, context.target.y
        self.laws_ = [ConstraintLaw(x + y == 3, target=x), ConstraintLaw(x - y == 1, target=y)]


def test_a_system_of_constraints_alone_is_static():
    """Test that a system whose laws are all constraints is solved once, as a static system."""
    system = System(
        elements=[Element('pair', values={'x': 0.0, 'y': 0.0})],
        interactions=[BalanceInteraction('balance')],
    ).prepare()
    trajectory = system.simulate(until=1.0)
    assert system.form_ == 'static solution'
    assert trajectory.times.tolist() == [0.0]
    assert [float(trajectory.get_values(name)[0]) for name in ('pair.x', 'pair.y')] == [2.0, 1.0]


def test_points_placed_in_a_space_with_no_laws_record_one_frame():
    """Test that a system with no laws is static, so it records its one frame for any `until` and `dt`."""
    square = ContinuousSpace(extent=(10.0, 10.0))
    disks = Element('disks', values={'x': [[1.0, 2.0], [3.0, 4.0]]}, size=2, space=square)
    system = System(elements=[disks], space=square).prepare()
    trajectory = system.simulate(until=1.0, dt=0.01)
    assert system.form_ == 'static solution'
    assert trajectory.times.tolist() == [0.0]
    assert trajectory.get_values('disks.x').tolist() == [[[1.0, 2.0], [3.0, 4.0]]]
