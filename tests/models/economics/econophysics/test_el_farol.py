"""Test the `_el_farol` module of `skcomplex.models.economics.econophysics`."""

import numpy as np

from skcomplex.engine.runtime import Simulation
from skcomplex.models import describe
from skcomplex.models.economics.econophysics import (
    el_farol,
)


def test_el_farol_attendance_near_capacity():
    """With 100 patrons and capacity 60, the mean attendance after the transient lies within 5 of 60."""
    system = el_farol(100, capacity=60, random_state=0)
    attendance = (
        Simulation(system, random_state=0, observables=describe('el_farol').observables_of(system))
        .prepare()
        .run(500)
        .trajectory_
    )
    assert abs(np.mean(attendance.values('attendance')[100:]) - 60) < 5
