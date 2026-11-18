"""Test the `_el_farol` module of `skcomplex.systems.economics.econophysics`."""

import numpy as np

from skcomplex.systems.economics.econophysics import (
    ElFarolSystem,
)


def test_el_farol_attendance_near_capacity():
    """With 100 patrons and capacity 60, the mean attendance after the transient lies within 5 of 60."""
    system = ElFarolSystem(100, capacity=60, random_state=0).prepare().system_
    attendance = system.prepare().simulate(
        until=500,
        random_state=0,
        observables=ElFarolSystem().prepare().observables_,
    )
    assert abs(np.mean(attendance.get_values('attendance')[100:]) - 60) < 5
