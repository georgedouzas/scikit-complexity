"""Test the `_stigmergy` module of `skcomplex.systems.biology.ecology`."""

from skcomplex.systems.biology.ecology import (
    StigmergicPredatorPreySystem,
)

from ..conftest import _run_until


def test_stigmergic_marks_follow_emotions():
    """Test that only animals with emotions mark the ground."""
    marks = {}
    for emotions in (True, False):
        system = (
            StigmergicPredatorPreySystem(20, 20, plants=100, prey=40, predators=10, emotions=emotions, random_state=0)
            .prepare()
            .system_
        )
        trajectory = _run_until(system, 10, random_state=0)
        marks[emotions] = trajectory.get_values('ground/marks.prey_happiness')[-1].sum()
        assert trajectory.get_values('life/prey.count')[-1] > 0
    assert marks[True] > 0
    assert marks[False] == 0


def test_stigmergic_prey_do_not_outlive_their_food():
    """Test that prey starve within their starvation time once the plants are gone, offspring included."""
    trajectory = _run_until(StigmergicPredatorPreySystem(random_state=1), 60, random_state=1)
    plants = [len(trajectory.get_agents('life/plants', frame).ids) for frame in range(len(trajectory.times))]
    prey = [len(trajectory.get_agents('life/prey', frame).ids) for frame in range(len(trajectory.times))]
    gone = plants.index(0)
    assert prey[gone + 12] == 0
