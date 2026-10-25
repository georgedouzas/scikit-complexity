"""Compile particles: pair interactions within a cutoff and the boundaries of the continuous space they move in."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import cast

from ....core.base import (
    BaseInteraction,
    BaseLaw,
    Expression,
    ValidationError,
    VariableKind,
    components_of,
    declarations_of,
)
from ....core.laws import BoundaryLaw, Symbols, either, mod, where
from ....core.problems import Source
from ....core.problems._kernels import PairSearch
from ....core.space import ContinuousSpace
from .._compilation import BlockLaws, Compilation
from .._declarations import Block, is_interaction
from .._pipeline import BasePass
from .._specs import KernelSpec, resolve_role
from .._tree import Node


def _space_period(node: Node) -> float | None:
    """Return the side of the periodic box that a pair interaction measures minimum image distances in, or `None`."""
    space = node.space
    if not isinstance(space, ContinuousSpace) or space.boundary != 'periodic':
        return None
    if len(set(space.extent)) > 1:
        error_msg = (
            f'Pair interaction {node.path} lives in the periodic box {space.extent}, whose sides differ. Pair searches '
            'measure minimum image distances in boxes with equal sides.'
        )
        raise ValidationError(error_msg)
    return space.extent[0]


def _pair_spec(block: Block, elements: dict[str, Block], time: Expression) -> KernelSpec:
    """Return the kernel spec of a pair interaction that is evaluated once per pair within its cutoff."""
    node = block.node
    interaction: BaseInteraction = node.component
    first, second = resolve_role(node, 'first', elements), resolve_role(node, 'second', elements)
    position = getattr(type(interaction), 'position', 'x')
    for bound in (first, second):
        entry = bound.entries.get(position)
        if entry is None or not entry.member_shape:
            error_msg = f'Pair interaction {node.path} needs the vector state {position!r} in {bound.node.path}.'
            raise ValidationError(error_msg)
    parameters, references = block.namespace(Source.SINGLE)
    first_namespace, first_references = first.namespace(Source.FIRST)
    second_namespace, second_references = second.namespace(Source.SECOND, alias='_pair')
    references.update(first_references)
    references.update(second_references)
    period = getattr(interaction, 'period', None)
    period = period if period is not None else _space_period(node)
    pairs = PairSearch(first.entries[position], second.entries[position], block.entries['cutoff'], period)
    namespace = Symbols(t=time, **vars(parameters), first=first_namespace, second=second_namespace)
    paths: dict[str, str | tuple[str, ...]] = {'first': first.node.path, 'second': second.node.path}
    return KernelSpec(node.path, namespace, references, 0, pairs, roles=paths)


def _space_boundary_laws(block: Block, spec: KernelSpec) -> list[BaseLaw]:
    """Return the laws by which the boundary of a continuous space acts on the position states of a component.

    A periodic boundary wraps every position component into the box. A reflecting boundary mirrors a position that
    crossed a wall back into the box and reverses the velocity component across the wall, where the velocity is the
    state declared as the rate of the position, or the first derivative of a second-order position.
    """
    space = block.node.space
    if is_interaction(block.node) or not isinstance(space, ContinuousSpace) or space.boundary == 'open':
        return []
    declarations = declarations_of(block.node.component_type, VariableKind.STATE)
    laws: list[BaseLaw] = []
    for name, declaration in declarations.items():
        if not declaration.position:
            continue
        velocity_name = next(
            (other for other, item in declarations.items() if item.velocity_of == name),
            f'{name}_d1' if declaration.order > 1 else None,
        )
        positions = components_of(getattr(spec.namespace, name))
        velocities = components_of(getattr(spec.namespace, velocity_name)) if velocity_name else [None] * len(positions)
        for position, velocity, side in zip(positions, velocities, space.extent, strict=True):
            if space.boundary == 'periodic':
                laws.append(BoundaryLaw(position, mod(position, side)))
                continue
            below, above = position < 0, position > side
            laws.append(BoundaryLaw(position, where(below, -position, where(above, 2 * side - position, position))))
            if velocity is not None:
                laws.append(BoundaryLaw(velocity, where(either(below, above), -velocity, velocity)))
    return laws


class BoundaryLawsPass(BasePass):
    """Pass that adds the laws by which the boundary of a continuous space acts on the positions of components."""

    reads = ('collected',)
    writes = ('collected',)

    def run(self, compilation: Compilation) -> None:
        """Add the boundary laws of continuous spaces.

        Args:
            compilation:
                The compilation state.
        """
        compilation.collected = [
            (spec, [*laws, *_space_boundary_laws(block, spec)], block)
            for spec, laws, block in cast('BlockLaws', compilation.collected)
        ]


PASSES = (BoundaryLawsPass(),)
BINDINGS = {'pairs': _pair_spec}
