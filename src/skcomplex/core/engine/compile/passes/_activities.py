"""Compile the activities of update laws into the recorders of their events."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import replace

import numpy as np
from sage.all__sagemath_symbolics import SR
from sage.modules.free_module_element import FreeModuleElement

from ....components.laws import UpdateLaw
from ....components.laws._local_rule import _LocalRuleLaw
from ....components.spaces import (
    AdjacencyNeighborhood,
    BaseNeighborhood,
    BaseSpace,
    LatticeSpace,
    NetworkSpace,
    RadiusNeighborhood,
)
from ....errors import ValidationError
from ....types import Indices
from ...enums import ReadKind
from ...problems._activity_recorder import _ActivityRecorder
from ...problems._expression_printer import ExpressionPrinter
from ...problems._kernel_spec import KernelSpec
from .._base import BasePass
from .._compilation import Compilation, _Declarations


class ActivitiesPass(BasePass):
    """A pass that adds the recorders of the events of activity to the problem.

    It compiles the activity of every update law that gives one, such as the burning sites of a forest fire, into a
    function of the values of its members, and finds the nearest neighbours of their space, which join the active
    members into events.

    Read more in the user guide, under Extend the library.

    See Also:
        LatticeProblemPass: Adds the steps of the laws of a lattice.

    Examples:
        >>> from skcomplex.core.engine.compile._pass_compiler import PassCompiler
        >>> from skcomplex.systems.physics.criticality import ForestFireSystem
        >>> compilation = PassCompiler().prepare(ForestFireSystem((4, 4)).prepare().system_).run().compilation_
        >>> [recorder.label for recorder in compilation.problem.activities]
        ['fire']
    """

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('problem', 'reduced', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('problem',)

    def _run(self, inputs: object = None) -> None:
        """Add the recorders of the activities to the problem."""
        compilation = self.compilation_
        declarations = Compilation._get_required(compilation.layout, 'layout')
        problem = Compilation._get_required(compilation.problem, 'problem')
        recorders = tuple(
            self._get_recorder(spec, law, declarations)
            for spec, laws in Compilation._get_required(compilation.reduced, 'reduced')
            for law in laws
            if isinstance(law, UpdateLaw | _LocalRuleLaw) and law.activity is not None
        )
        if recorders:
            compilation.problem = replace(problem, activities=recorders)

    @staticmethod
    def _get_nearest(space: BaseSpace) -> BaseNeighborhood:
        """Return the nearest neighbours of a space of sites.

        They are the von Neumann neighbours of a square lattice, the six neighbours of a triangular one, and the
        neighbours along the edges of a network.
        """
        if isinstance(space, LatticeSpace):
            return RadiusNeighborhood(1, metric='manhattan' if space.geometry == 'square' else 'hexagonal')
        if isinstance(space, NetworkSpace):
            return AdjacencyNeighborhood()
        return space._get_default_neighborhood()

    @staticmethod
    def _get_neighbors(owner: str, declarations: _Declarations) -> Indices | None:
        """Return the nearest neighbours of every member of a group, or `None` for members of no space."""
        block = declarations.blocks.get(owner)
        space = block.node.space if block is not None else None
        if not isinstance(space, BaseSpace) or not space._has_sites():
            return None
        return space._get_index_table(ActivitiesPass._get_nearest(space))

    @staticmethod
    def _get_recorder(
        spec: KernelSpec,
        law: UpdateLaw | _LocalRuleLaw,
        declarations: _Declarations,
    ) -> _ActivityRecorder:
        """Return the recorder of the events of the activity of an update law.

        Args:
            spec:
                What the laws of the component of the law read.

            law:
                The update law, or the local rule of one.

            declarations:
                The values every component declares.

        Returns:
            recorder:
                The recorder of the events of the law.

        Raises:
            ValidationError:
                The activity reads a value that is no value of the members the law runs for or of its component.
        """
        label = law.label
        target = law._get_targets()[0]
        target = target[0] if isinstance(target, FreeModuleElement) else target
        members = spec.references.get(target)
        if members is None:
            error_msg = f'The update law {label!r} of {spec.owner} has an activity, but its target is no state.'
            raise ValidationError(error_msg)
        size = int(members.flat_indices.size)
        expression = SR(law.activity)
        time = declarations.time
        arguments = sorted(set(expression.variables()) - {time}, key=str)
        reads: list[tuple[bool, Indices]] = []
        for item in arguments:
            reference = spec.references.get(item)
            if reference is None or reference.source not in {ReadKind.MEMBER, ReadKind.SINGLE}:
                error_msg = (
                    f'The activity of the update law {label!r} of {spec.owner} reads {item}, which is no value of the '
                    'member the law runs for. Give an expression of its values and parameters.'
                )
                raise ValidationError(error_msg)
            rows = reference.flat_indices
            reads.append((reference.is_state, rows if rows.size == size else np.repeat(rows[:1], size)))
        function = ExpressionPrinter(time=time, compiled=True).prepare(arguments).run([expression]).code_
        if not callable(function):
            error_msg = f'A compiled printer returns a function, not a {type(function).__name__}.'
            raise TypeError(error_msg)
        owner = members.entry.name.rpartition('.')[0]
        return _ActivityRecorder(label, function, tuple(reads), size, ActivitiesPass._get_neighbors(owner, declarations))
