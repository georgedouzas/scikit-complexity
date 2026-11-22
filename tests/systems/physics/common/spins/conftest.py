"""Share the spin holder and the runs of the spin sampling tests."""

import numpy as np
import numpy.typing as npt

from skcomplex.core.components.elements import Element
from skcomplex.core.components.interactions import BaseInteraction
from skcomplex.core.components.spaces import LatticeSpace
from skcomplex.core.engine.solvers import MetropolisSolver
from skcomplex.core.systems import System

SITES = {'spin': 1.0, 'color': 0.0, 'theta': 0.0, 'J': 1.0, 'h': 0.0, 'temperature': 1.0}


def _run(
    interaction: BaseInteraction,
    state: str,
    values: dict[str, object],
    solver: MetropolisSolver | None = None,
) -> npt.NDArray[np.float64]:
    """Return the values of a state of a 12 by 12 lattice of sites after 100 sweeps, flipping it by default."""
    sites = Element('sites', values={**SITES, **values}, space=LatticeSpace((12, 12)))
    solver = solver or MetropolisSolver(moves={f'sites.{state}': 'flip'})
    system = System(elements=[sites], interactions=[interaction], solver=solver).prepare()
    return np.asarray(system.simulate(until=100, random_state=0).get_values(f'sites.{state}'))
