"""Advance lattice problems by local rules, relaxations, and Monte Carlo sweeps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar, cast

from ...core.laws import BoundaryLaw, DriveLaw, ExtremalLaw, LocalRuleLaw, RelaxationLaw, SampleLaw, UpdateLaw
from ...core.spaces import DiscreteTime, LatticeSpace, NetworkSpace
from ...core.types import Vector
from ..compile._lattice_steps import CompiledExtremal, CompiledRelaxation, LatticeStep
from ..problems import Form, OrderKind, SupportSpec
from ._stepper import Solver


class LatticeSolver(Solver):
    """Solver of lattice dynamics that applies every local rule, drive, relaxation, and extremal law once per step.

    Steps apply in the order the laws are declared, and the boundary laws apply after every step. Relaxations record
    their size, duration, and area as events, and extremal laws record the site and the value of every application.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.compile import compile_system
        >>> from skcomplex.models.computer_science.automata import elementary_automaton
        >>> from skcomplex.engine.solvers import LatticeSolver
        >>> problem = compile_system(elementary_automaton(30, size=11))
        >>> stepper = LatticeSolver().prepare(problem, np.random.default_rng(0))
        >>> stepper.run(0.0, problem.initial_state, problem.initial_parameters, 1.0).astype(int).tolist()
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    """

    supports: ClassVar[SupportSpec] = SupportSpec(
        time={DiscreteTime},
        space={LatticeSpace, NetworkSpace},
        laws={LocalRuleLaw, DriveLaw, RelaxationLaw, ExtremalLaw, UpdateLaw, BoundaryLaw},
        orders={OrderKind.INTEGER},
    )

    form: ClassVar[Form] = Form.LATTICE

    def _advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
        """Apply the lattice laws once per integer time step from `t` up to `until`.

        Args:
            t:
                The current time.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The time to advance to.

        Returns:
            state:
                The state vector after the last whole step.
        """
        state = x.copy()
        for step in range(round(t), math.floor(until + 1e-9)):
            for item in cast('tuple[LatticeStep, ...]', self.problem_.lattice):
                if isinstance(item, CompiledRelaxation):
                    record = item.apply(state, p, self.rng_, float(step + 1))
                    if record is not None:
                        self.locator_.records.append(record)
                elif isinstance(item, CompiledExtremal):
                    self.locator_.records.extend(item.apply(state, p, self.rng_, float(step + 1)))
                else:
                    item.apply(state, p, self.rng_)
            state = self._bounded(float(step + 1), state, p)
        return state


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
