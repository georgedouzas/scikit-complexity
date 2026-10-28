"""Solver of Monte Carlo sampling that sweeps the lattice once per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

from ...core.laws import BoundaryLaw, SampleLaw
from ...core.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ..problems import Form, OrderKind, SupportSpec
from ._lattice import LatticeSolver


class MonteCarloSolver(LatticeSolver):
    """Solver of Monte Carlo sampling that sweeps the lattice once per unit of time.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.physics.statistical_mechanics import ising
        >>> from skcomplex.engine.solvers import MonteCarloSolver
        >>> problem = compile_system(ising(size=8, temperature=1.0))
        >>> stepper = MonteCarloSolver().prepare(problem, np.random.default_rng(0))
        >>> state = stepper.run(0.0, problem.initial_state, problem.initial_parameters, 50.0)
        >>> bool(abs(state.mean()) > 0.9)
        True
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={DiscreteTime},
        space={LatticeSpace, NetworkSpace},
        laws={SampleLaw, BoundaryLaw},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.MONTE_CARLO
