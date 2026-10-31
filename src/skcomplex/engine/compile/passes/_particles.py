"""Compile particles: pair interactions within a cutoff and the boundaries of the continuous space they move in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import SR

from ....core.laws import BaseLaw, BoundaryLaw
from ....core.laws.utils import either, mod, where
from ....core.spaces import ContinuousSpace
from ....types import VariableKind
from ..._utils import components_of, declarations_of
from ...problems import BlockSpec, KernelSpec
from .._base import BasePass
from .._compilation import Compilation


class BoundaryLawsPass(BasePass):
    """Pass that adds the laws by which the boundary of a continuous space acts on the positions of components."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('collected',)

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('collected',)

    def _run(self, inputs: object = None) -> None:
        """Add the boundary laws of continuous spaces."""
        compilation = self.compilation_
        compilation.collected = [
            (spec, [*laws, *self._space_boundary_laws(block, spec)], block)
            for spec, laws, block in Compilation._required(compilation.collected, 'collected')
        ]

    @staticmethod
    def _space_boundary_laws(block: BlockSpec, spec: KernelSpec) -> list[BaseLaw]:
        """Return the laws by which the boundary of a continuous space acts on the position states of a component.

        A periodic boundary wraps every position component into the box. A reflecting boundary mirrors a position that
        crossed a wall back into the box and reverses the velocity component across the wall, where the velocity is the
        first derivative of a position of order 2.
        """
        space = block.node.space
        if block.node.interaction or not isinstance(space, ContinuousSpace) or space.boundary == 'open':
            return []
        declarations = declarations_of(block.node.component_type, VariableKind.STATE)
        laws: list[BaseLaw] = []
        for name, declaration in declarations.items():
            if not declaration.position:
                continue
            velocity_name = f'{name}_d1' if declaration.order > 1 else None
            positions = [SR(item) for item in components_of(getattr(spec.namespace, name))]
            velocities = (
                [SR(item) for item in components_of(getattr(spec.namespace, velocity_name))]
                if velocity_name
                else [None] * len(positions)
            )
            for position, velocity, side in zip(positions, velocities, space.extent_, strict=True):
                if space.boundary == 'periodic':
                    laws.append(BoundaryLaw(position, mod(position, side)))
                    continue
                below, above = position < 0, position > side
                laws.append(BoundaryLaw(position, where(below, -position, where(above, 2 * side - position, position))))
                if velocity is not None:
                    laws.append(BoundaryLaw(velocity, where(either(below, above), -velocity, velocity)))
        return laws
