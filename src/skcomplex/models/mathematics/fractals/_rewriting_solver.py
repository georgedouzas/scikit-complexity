"""Advance rewriting systems one generation per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.spaces import CayleySpace, ContinuousSpace, DiscreteTime, LatticeSpace, MeshSpace, NetworkSpace
from ....engine.solvers import BaseSolver
from ....errors import SolverError
from ....specs import RunStateSpec, StructureStateSpec, SupportSpec
from ....types import Array, OrderKind
from ._patch_structure_spec import PatchStructureSpec
from ._production_law import ProductionLaw
from ._rewriting_problem_spec import REWRITING, RewritingProblemSpec
from ._substitution_law import SubstitutionLaw
from ._word_structure_spec import WordStructureSpec


class RewritingSolver(BaseSolver[RewritingProblemSpec]):
    """Solver of rewriting systems, which rewrites every structure in parallel once per generation.

    The structures live next to the state vector, whose components are their symbol, prototile, or label counts and
    sizes. Stochastic productions draw from the simulation's random stream.

    Examples:
        >>> from skcomplex.core.spaces import DiscreteTime
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> from skcomplex.models.mathematics.fractals import ProductionLaw
        >>> from skcomplex.models.mathematics.fractals import RewritingSolver
        >>> from skcomplex.models.mathematics.fractals import WordElement
        >>> class AlgaeElement(WordElement):
        ...     def _get_laws(self, namespace):
        ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
        >>> system = System(elements=[AlgaeElement('algae', axiom='A')], time=DiscreteTime())
        >>> from skcomplex.models.mathematics.fractals import RewritingCompiler
        >>> compiler = RewritingCompiler()
        >>> simulation = Simulation(system, compiler=compiler, solver=RewritingSolver())
        >>> simulation = simulation.check().prepare()
        >>> trajectory = simulation.run(until=5).trajectory_
        >>> trajectory.values('algae.size').tolist()
        [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    """

    @classmethod
    def _get_support(cls) -> SupportSpec:
        """Return the times, spaces, laws, noise, and orders the solver solves."""
        return SupportSpec(
            time={DiscreteTime},
            space={None, LatticeSpace, NetworkSpace, ContinuousSpace, MeshSpace, CayleySpace},
            laws={ProductionLaw, SubstitutionLaw},
            orders=frozenset({OrderKind.INTEGER}),
        )

    @classmethod
    def _get_form(cls) -> str:
        """Return the form the solver solves."""
        return REWRITING

    @classmethod
    def _is_stepwise(cls) -> bool:
        """Return that the solver rewrites once per whole generation."""
        return True

    def _prepare(self, problem: RewritingProblemSpec | None) -> None:
        """Prepare the structures at generation 0.

        Args:
            problem:
                The compiled problem.

        Raises:
            SolverError:
                No problem is given.
        """
        if problem is None:
            error_msg = 'The rewriting solver prepares for a compiled rewriting problem, but got none.'
            raise SolverError(error_msg)
        super()._prepare(problem)
        self.systems_ = problem.rewriting
        self.structures_: dict[str, WordStructureSpec | PatchStructureSpec] = {
            item.path: item.prepare(problem).values_ for item in self.systems_
        }
        self._collect()

    def _state(self) -> Array:
        """Return the state vector of the counts and sizes of the current structures."""
        counts = [value for item in self.systems_ for value in item.counts(self.structures_[item.path])]
        return np.asarray(counts, dtype=float)

    def _advance(self, t: float, x: Array, p: Array, until: float) -> Array:
        """Rewrite every structure once per generation from `t` to `until`.

        Args:
            t:
                The current generation.

            x:
                The current state vector.

            p:
                The parameter vector.

            until:
                The generation to advance to.

        Returns:
            state:
                The counts and sizes at generation `until`.
        """
        for generation in range(round(until - t)):
            state = RunStateSpec(t + generation, x, p, self.rng_)
            self.structures_ = {item.path: item.run(state).values_ for item in self.systems_}
        return self._state()

    def _collect(self) -> None:
        """Set the snapshot of the current structures, which later generations do not change, and their measures."""
        reporters: dict[str, float] = {}
        for path, structure in self.structures_.items():
            if structure.has_geometry:
                reporters[f'{path}.length'] = structure.length()
                reporters[f'{path}.area'] = structure.area()
        self.events_ = []
        self.snapshot_ = StructureStateSpec(dict(self.structures_), reporters)
