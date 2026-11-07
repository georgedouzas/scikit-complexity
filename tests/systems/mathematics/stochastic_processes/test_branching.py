"""Test the `_branching` module of `skcomplex.systems.mathematics.stochastic_processes`."""

import numpy as np
import pytest

from skcomplex.systems.mathematics.stochastic_processes import (
    GaltonWatsonSystem,
)

from ..conftest import _run


def test_galton_watson_grows_geometrically_on_average():
    """Test that the mean generation size is `Z0 m^n`, and that subcritical processes die out."""
    sizes = [
        _run(GaltonWatsonSystem(50, m=1.2), 5, random_state=seed).values('population.count')[-1] for seed in range(30)
    ]
    assert np.mean(sizes) == pytest.approx(50 * 1.2**5, rel=0.15)
    assert _run(GaltonWatsonSystem(10, m=0.5), 30).values('population.count')[-1] == 0
