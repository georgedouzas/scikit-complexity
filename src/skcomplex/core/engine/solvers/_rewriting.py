"""Advance rewriting systems one generation per step."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ...components.laws import ProductionLaw, SubstitutionLaw
from ...components.spaces import CayleySpace, ContinuousSpace, DiscreteTime, LatticeSpace, MeshSpace, NetworkSpace
from ...errors import SolverError
from ...specs import PatchStructureSpec, RunStateSpec, StructureStateSpec, SupportSpec, WordStructureSpec
from ...types import Array, FormKind, OrderKind
from ..problems._patch_kernel import PatchKernel
from ..problems._rewriting_problem_spec import RewritingProblemSpec
from ..problems._word_kernel import WordKernel
from ._base import BaseSolver


class RewritingSolver(BaseSolver[RewritingProblemSpec]):
    """Solver of rewriting systems.

    It rewrites every structure in parallel once per generation. The structures live next to the state vector, whose
    components are their symbol, prototile, or label counts and sizes. Stochastic productions draw from the simulation's
    random stream.

    Examples:
        >>> from skcomplex.core.components.spaces import DiscreteTime
        >>> from skcomplex.core.components.systems import CompositeSystem
        >>> from skcomplex.core.engine.runtime import simulate
        >>> from skcomplex.core.components.laws import ProductionLaw
        >>> from skcomplex.core.engine.solvers import RewritingSolver
        >>> from skcomplex.core.components.elements import WordElement
        >>> class AlgaeElement(WordElement):
        ...     def _get_laws(self, namespace):
        ...         return [ProductionLaw('A', 'AB'), ProductionLaw('B', 'A')]
        >>> system = CompositeSystem(elements=[AlgaeElement('algae', axiom='A')], time=DiscreteTime())
        >>> trajectory = simulate(system, until=5, solver=RewritingSolver())
        >>> trajectory.get_values('algae.size').tolist()
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
    def _get_form(cls) -> FormKind:
        """Return the form the solver solves."""
        return FormKind.REWRITING

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

    def _get_state(self) -> Array:
        """Return the state vector of the counts and sizes of the current structures."""
        counts: list[float] = []
        for item in self.systems_:
            structure = self.structures_[item.path]
            if isinstance(item, WordKernel) and isinstance(structure, WordStructureSpec):
                counts += item.get_counts(structure)
            if isinstance(item, PatchKernel) and isinstance(structure, PatchStructureSpec):
                counts += item.get_counts(structure)
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
        return self._get_state()

    def _collect(self) -> None:
        """Set the snapshot of the current structures and their measures.

        Later generations do not change the snapshot.
        """
        reporters: dict[str, float] = {}
        for path, structure in self.structures_.items():
            if structure.has_geometry:
                reporters[f'{path}.length'] = structure.get_length()
                reporters[f'{path}.area'] = structure.get_area()
        self.events_ = []
        self.snapshot_ = StructureStateSpec(dict(self.structures_), reporters)
