"""Solver of Monte Carlo sampling that sweeps the lattice once per unit of time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


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
        >>> from skcomplex.engine.solvers import MonteCarloSolver, Step
        >>> problem = compile_system(ising(size=8, temperature=1.0))
        >>> stepper = MonteCarloSolver(random_state=0).check().prepare(problem)
        >>> state = stepper.run(Step(0.0, problem.initial_state, problem.initial_parameters, 50.0))
        >>> bool(abs(state.mean()) > 0.9)
        True
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={LatticeSpace, NetworkSpace},
            laws={SampleLaw, BoundaryLaw},
            orders={OrderKind.INTEGER},
        )

    @classmethod
    def _get_form(cls) -> Form:
        """Return the form the solver solves."""
        return Form.MONTE_CARLO
