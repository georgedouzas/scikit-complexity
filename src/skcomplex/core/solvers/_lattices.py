"""Advance lattice problems by local rules, relaxations, and Monte Carlo sweeps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import ClassVar

from ..base import Form
from ..compile import CompiledExtremal, CompiledRelaxation, Vector
from ._base import Stepper, register_solver


class LatticeStepper(Stepper):
    """Stepper of lattice dynamics that applies every local rule, drive, relaxation, and extremal law once per step.

    Steps apply in the order the laws are declared. Relaxations record their size, duration, and area as events, and
    extremal laws record the site and the value of every application.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.models.computer_science.automata import elementary_automaton
        >>> from skcomplex.core.solvers import LatticeStepper
        >>> problem = compile_system(elementary_automaton(30, size=11))
        >>> stepper = LatticeStepper()
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 1.0).astype(int).tolist()
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    """

    form: ClassVar[Form] = Form.LATTICE

    def advance(self, t: float, x: Vector, p: Vector, until: float) -> Vector:
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
            for item in self.problem_.lattice:
                if isinstance(item, CompiledRelaxation):
                    record = item.apply(state, p, self.rng_, float(step + 1))
                    if record is not None:
                        self.locator_.records.append(record)
                elif isinstance(item, CompiledExtremal):
                    self.locator_.records.extend(item.apply(state, p, self.rng_, float(step + 1)))
                else:
                    item.apply(state, p, self.rng_)
        return state


class MonteCarlo(LatticeStepper):
    """Stepper of Monte Carlo sampling that sweeps the lattice once per unit of time.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.core.compile import compile_system
        >>> from skcomplex.models.physics.statistical_mechanics import ising
        >>> from skcomplex.core.solvers import MonteCarlo
        >>> problem = compile_system(ising(size=8, temperature=1.0))
        >>> stepper = MonteCarlo()
        >>> stepper.initialize(problem, np.random.default_rng(0))
        >>> state = stepper.advance(0.0, problem.initial_state, problem.initial_parameters, 50.0)
        >>> bool(abs(state.mean()) > 0.9)
        True
    """

    form: ClassVar[Form] = Form.MONTE_CARLO


register_solver('lattice', LatticeStepper, default=True)
register_solver('monte-carlo', MonteCarlo, default=True)
