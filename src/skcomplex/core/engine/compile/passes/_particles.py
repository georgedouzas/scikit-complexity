"""Compile particles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from sage.all__sagemath_symbolics import SR

from ....components.laws import BaseLaw
from ....components.laws._boundary import _BoundaryLaw
from ....components.laws.utils import either, mod, where
from ....components.spaces import ContinuousSpace
from ....enums import VariableKind
from ..._utils import components_of, declarations_of
from ...problems._block_spec import BlockSpec
from ...problems._kernel_spec import KernelSpec
from .._base import BasePass
from .._compilation import Compilation


class BoundaryLawsPass(BasePass):
    """A pass that adds the laws of the boundary of a continuous space.

    By these laws the boundary acts on the positions of components.
    """

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
            (spec, [*laws, *self._get_space_boundary_laws(block, spec)], block)
            for spec, laws, block in Compilation._get_required(compilation.collected, 'collected')
        ]

    @staticmethod
    def _get_space_boundary_laws(block: BlockSpec, spec: KernelSpec) -> list[BaseLaw]:
        """Return the laws by which a continuous space boundary acts on a component.

        It acts on the position states of the component: a vector position, or a scalar value named after an axis of
        the space, such as `x` and `y` of `Element('walkers', values={'x': [1.0, 9.0], 'y': [5.0, 5.0]}, size=2)`. A
        periodic boundary wraps every position component into the box. A reflecting boundary mirrors a position that
        crossed a wall back into the box and reverses the velocity component across the wall, where the velocity is
        the first derivative of a position of order 2. Agents keep their own positions in the space, which wraps or
        bounds them, so they take no law.
        """
        space = block.node.space
        if (
            block.node.interaction
            or block.node.is_agent
            or not isinstance(space, ContinuousSpace)
            or space.boundary == 'open'
        ):
            return []
        declarations = declarations_of(block.node.component_type, VariableKind.STATE)
        laws: list[BaseLaw] = []
        for name, declaration in declarations.items():
            if declaration.is_place:
                sides = space.extent_
            elif declaration.role == 'position' and name in space.axes_:
                sides = (space.extent_[space.axes_.index(name)],)
            else:
                continue
            velocity_name = f'{name}_d1' if declaration.order > 1 else None
            positions = [SR(item) for item in components_of(getattr(spec.namespace, name))]
            velocities = (
                [SR(item) for item in components_of(getattr(spec.namespace, velocity_name))]
                if velocity_name
                else [None] * len(positions)
            )
            for position, velocity, side in zip(positions, velocities, sides, strict=True):
                if space.boundary == 'periodic':
                    laws.append(_BoundaryLaw(position, mod(position, side)))
                    continue
                below, above = position < 0, position > side
                laws.append(
                    _BoundaryLaw(position, where(below, -position, where(above, 2 * side - position, position))),
                )
                if velocity is not None:
                    laws.append(_BoundaryLaw(velocity, where(either(below, above), -velocity, velocity)))
        return laws
