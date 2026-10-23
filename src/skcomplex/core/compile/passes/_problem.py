"""Assemble the problem of a compilation from its layout, its numeric functions, and its compiled events."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, cast

import numpy as np

from ...base import Form, VariableKind, VisualSpec, declarations_of
from ...components import Group
from .._compilation import Compilation, Laws
from .._declarations import Block, Declarations, is_interaction
from .._layout import Layout
from .._pipeline import Pass
from .._problem import CompiledElement, CompiledInteraction, MarkovMatrix, Problem
from .._reductions import ReductionPass
from .._specs import KernelSpec
from .._tree import Node
from ._events import breakpoints_in, compile_delays, compile_events, hidden_events


def _companions(blocks: list[Block]) -> dict[str, tuple[str, str, int]]:
    """Return every companion state by name, with its kind, the name of its state, and its index."""
    companions = {}
    for block in blocks:
        if is_interaction(block.node) or block.node.is_agent:
            continue
        declarations = declarations_of(block.node.component_type, VariableKind.STATE)
        for variable, entry in block.entries.items():
            declaration = declarations.get(variable)
            if declaration is None or (declaration.derivative_of is None and declaration.lag_of is None):
                continue
            kind, base = (
                ('derivative', declaration.derivative_of) if declaration.derivative_of else ('lag', declaration.lag_of)
            )
            companions[entry.name] = (kind, f'{block.node.path}.{base}', declaration.index)
    return companions


def _history(blocks: list[Block]) -> dict[str, Any]:
    """Return the history every state declares, by the name of its layout entry."""
    history = {}
    for block in blocks:
        if is_interaction(block.node) or block.node.is_agent:
            continue
        declarations = declarations_of(block.node.component_type, VariableKind.STATE)
        for variable, entry in block.entries.items():
            declaration = declarations.get(variable)
            if declaration is not None and declaration.history is not None:
                history[entry.name] = declaration.history
    return history


def visual_of(node: Node) -> VisualSpec:
    """Return the visual of an element, or a group's own visual when it sets one and its default otherwise."""
    if isinstance(node.component, Group):
        return node.component.visual if node.component.visual is not None else node.component.default_visual()
    declared: VisualSpec = node.component_type.visual
    return declared


def _compiled_elements(blocks: list[Block]) -> tuple[CompiledElement, ...]:
    """Return the elements and groups that are not agents with the dotted names of their states and parameters."""
    return tuple(
        CompiledElement(
            path=block.node.path,
            element=block.node.component,
            states={
                name: f'{block.node.path}.{name}'
                for name in declarations_of(block.node.component_type, VariableKind.STATE)
            },
            parameters={
                name: f'{block.node.path}.{name}'
                for name in declarations_of(block.node.component_type, VariableKind.PARAMETER)
            },
            visual=visual_of(block.node),
            members=block.node.members,
        )
        for block in blocks
        if not is_interaction(block.node) and not block.node.is_agent
    )


def _compiled_interactions(
    blocks: list[Block],
    specs: list[tuple[KernelSpec, Block]],
) -> tuple[CompiledInteraction, ...]:
    """Return the interactions with the paths their roles bind to."""
    return tuple(
        CompiledInteraction(
            path=block.node.path,
            interaction=block.node.component,
            roles=next(spec.roles for spec, _ in specs if spec.owner == block.node.path),
        )
        for block in blocks
        if is_interaction(block.node)
    )


def _problem_of(compilation: Compilation) -> Problem:
    """Return the problem of the general parts of a compilation, which passes of formalisms add to."""
    declarations = cast('Declarations', compilation.layout)
    blocks = list(declarations.blocks.values())
    form = cast('Form', compilation.form)
    reduced = cast('Laws', compilation.reduced)
    reductions = cast('ReductionPass', compilation.reductions)
    time = declarations.time
    functions = cast('dict[str, Any]', compilation.functions)
    return Problem(
        form=form,
        states=Layout(declarations.states),
        parameters=Layout(declarations.parameters),
        initial_state=np.concatenate([np.zeros(0), *declarations.initial_states]),
        initial_parameters=np.concatenate([np.zeros(0), *declarations.initial_parameters]),
        time=time,
        symbols={entry.name: block.symbols[variable] for block in blocks for variable, entry in block.entries.items()},
        elements=_compiled_elements(blocks),
        interactions=_compiled_interactions(blocks, cast('list[tuple[KernelSpec, Block]]', compilation.specs)),
        system=compilation.system,
        companions=_companions(blocks),
        delays=compile_delays(compilation.delays, time),
        events=(
            *compile_events(compilation.events, time, reductions),
            *(hidden_events(reduced, time, reductions) if form is Form.ODE else ()),
        ),
        reductions=reductions.reductions,
        jacobian=compilation.jacobian,
        breakpoints=breakpoints_in(reduced, compilation.events, time),
        history=_history(blocks),
        matrices=cast('tuple[MarkovMatrix, ...]', compilation.matrices),
        **functions,
    )


class ProblemPass(Pass):
    """Pass that assembles the problem of the general parts of a compilation."""

    reads = (
        'layout',
        'specs',
        'reduced',
        'form',
        'functions',
        'jacobian',
        'matrices',
        'reductions',
        'events',
        'delays',
    )
    writes = ('problem',)

    def run(self, compilation: Compilation) -> None:
        """Assemble the problem.

        Args:
            compilation:
                The compilation state.
        """
        compilation.problem = _problem_of(compilation)
