"""Solver of lattice dynamics that applies every lattice law once per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from ...components.laws import UpdateLaw
from ...components.laws._boundary import _BoundaryLaw
from ...components.laws._local_rule import _LocalRuleLaw
from ...components.spaces import LatticeSpace, NetworkSpace
from ...components.spaces._discrete_time import _DiscreteTime
from ...enums import FormKind, OrderKind
from ...types import Array
from ..backends import SupportSpec
from ..problems import RunStateSpec
from ..problems._cascade_kernel import _CascadeRecordSpec
from ._solver import Solver


class LatticeSolver(Solver):
    """A solver that applies every law of a lattice once per step.

    The laws are local rules, drives, and relaxations, such as rule 30 that sets each cell from itself and its two
    neighbours. They apply in the order they are declared and read the time at the start of the step, as maps do. The
    boundary laws apply after every step at its end time, and a relaxation records the size, duration, area, and profile
    of every cascade as an event. An update law with an activity records every burst of activity connected in space and
    time as an event, read at the start of the run and at the end of every step. It runs cellular automata, the
    sandpiles of earthquakes, and the fires of a forest.

    Read more in the user guide, under Solvers.

    See Also:
        MonteCarloSolver: Samples the sites of a lattice at random.
        IterationSolver: Applies a map without a lattice.

    Examples:
        >>> from skcomplex.core.engine.solvers import LatticeSolver
        >>> from skcomplex.systems.computer_science.automata import ElementaryAutomatonSystem
        >>> system = ElementaryAutomatonSystem(30, size=11, solver=LatticeSolver()).prepare()
        >>> system.simulate(until=1).get_values('cells.alive')[-1].astype(int).tolist()
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={_DiscreteTime},
            space={LatticeSpace, NetworkSpace},
            laws={_LocalRuleLaw, UpdateLaw, _BoundaryLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.LATTICE

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
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
        self._record_activity(t, state, p, first=True)
        for step in range(round(t), math.floor(until + 1e-9)):
            current = RunStateSpec(float(step), state, p, self.rng_)
            for item in self.problem_.lattice:
                result = item.run(current).values_
                if isinstance(result, _CascadeRecordSpec):
                    self.locator_.records.append(result)
            state = self._get_bounded(float(step + 1), state, p)
            self._record_activity(float(step + 1), state, p)
        return state
