"""Read delayed states through hidden parameters, take events aside, and compile both with the breakpoints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
from collections.abc import Callable
from typing import cast

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.errors import ValidationError
from ....core.laws import BaseLaw, BoundaryLaw, CrossingLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....core.types import Vector
from ...problems import Entry, Reference, Source, numpy_function, symbol, vector
from .._compilation import Compilation, Laws
from .._declarations import Declarations
from .._pipeline import BasePass
from .._specs import KernelSpec

type Scalars = Callable[[float, Vector, Vector], list[float]]
RELATIONS = (operator.lt, operator.le, operator.gt, operator.ge)
_SWITCHES = frozenset({'sign', 'sgn', 'heaviside', 'unit_step'})


class DelaysPass(BasePass):
    """Pass that reads delayed states through hidden parameters and takes the events of the laws aside."""

    reads = ('ordered', 'layout')
    writes = ('delayed', 'delays', 'placeholders', 'events')

    def _run(self, compilation: Compilation) -> None:
        """Lower the delays of the ordered laws and take their events aside.

        Args:
            compilation:
                The compilation state.
        """
        declarations = cast('Declarations', compilation.layout)
        compilation.delayed = [
            (spec, self._with_delays(spec, laws, declarations, compilation))
            for spec, laws in cast('Laws', compilation.ordered)
        ]

    @staticmethod
    def _delay_minimum(
        spec: KernelSpec,
        tau: Expression,
        minimum: Expression | None,
        declarations: Declarations,
    ) -> float:
        """Return the smallest value of a delay: itself, its declared minimum, or its value at the parameters."""
        if SR(tau).is_numeric():
            return float(tau)
        if minimum is not None and SR(minimum).is_numeric():
            return float(minimum)
        symbols = sorted(SR(tau).variables(), key=str)
        references = [spec.references.get(item) for item in symbols]
        if symbols and all(reference is not None and not reference.is_state for reference in references):
            values = np.concatenate([np.zeros(0), *declarations.initial_parameters])
            columns = [reference.column_of(np.zeros(0), values) for reference in references if reference is not None]
            delays = numpy_function(declarations.time, symbols, [SR(tau)])(0.0, *columns)[0]
            return float(np.min(delays))
        error_msg = (
            f'A delay of {spec.owner} is the expression {tau}, whose smallest value is unknown. Give it as the third '
            'argument, `delayed(x, tau, minimum)`.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _substitute_delays(
        value: object,
        spec: KernelSpec,
        declarations: Declarations,
        compilation: Compilation,
    ) -> object:
        """Return a law value with every delayed state replaced by a hidden parameter that solvers fill from history."""
        if isinstance(value, FreeModuleElement):
            return vector(DelaysPass._substitute_delays(item, spec, declarations, compilation) for item in value)
        if not isinstance(value, Expression):
            return value
        replacements = {}
        for call in BasePass._calls_of(value, 'delayed'):
            arguments = call.operands()
            item, tau = arguments[0], arguments[1]
            reference = spec.references.get(item)
            if reference is None or not reference.is_state or reference.source not in {Source.SINGLE, Source.MEMBER}:
                error_msg = f'{spec.owner} delays {item}, which is not a state of an element or a group member.'
                raise ValidationError(error_msg)
            key = (id(spec), reference.entry.name, reference.column, str(tau))
            if key in compilation.placeholders:
                replacements[call] = compilation.placeholders[key]
                continue
            entry = reference.entry
            members = entry.members if reference.source is Source.MEMBER else None
            indices = (
                entry.start + np.arange(members) * entry.columns + reference.column
                if members is not None
                else np.asarray([entry.start + reference.column])
            )
            initial = np.concatenate([np.zeros(0), *declarations.initial_states])[indices]
            index = len(compilation.delays)
            hidden = Entry(
                f'__delayed_{index}',
                declarations.ends[False],
                (members,) if members is not None else (),
                RR,
                members,
            )
            declarations.parameters.append(hidden)
            declarations.ends[False] += hidden.size
            declarations.initial_parameters.append(np.asarray(initial, dtype=float))
            placeholder = symbol(f'_delayed_{index}')
            spec.references[placeholder] = Reference(hidden, False, 0, reference.source)
            compilation.delays.append(
                {
                    'entry': hidden,
                    'states': indices,
                    'tau': tau,
                    'minimum': DelaysPass._delay_minimum(
                        spec,
                        tau,
                        next(iter(arguments[2:]), None),
                        declarations,
                    ),
                    'spec': spec,
                    'symbol': placeholder,
                    'state': item,
                },
            )
            compilation.placeholders[key] = placeholder
            replacements[call] = placeholder
        return value.subs(replacements) if replacements else value

    @staticmethod
    def _with_delays(
        spec: KernelSpec,
        laws: list[BaseLaw],
        declarations: Declarations,
        compilation: Compilation,
    ) -> list[BaseLaw]:
        """Return laws with delayed states read through hidden parameters, keeping events aside in the compilation."""
        result: list[BaseLaw] = []
        for law in laws:
            if isinstance(law, CrossingLaw):
                guard = DelaysPass._substitute_delays(SR(law.guard), spec, declarations, compilation)
                reset = {
                    target: DelaysPass._substitute_delays(SR(value), spec, declarations, compilation)
                    for target, value in (law.reset or {}).items()
                }
                compilation.events.append(
                    (spec, CrossingLaw(guard, reset, law.direction, law.terminal, label=law.label)),
                )
            elif isinstance(law, JumpLaw):
                result.append(
                    JumpLaw(DelaysPass._substitute_delays(law.rate, spec, declarations, compilation), law.changes),
                )
            elif isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
                result.append(
                    BasePass._replaced(
                        law, value=DelaysPass._substitute_delays(law.value, spec, declarations, compilation),
                    ),
                )
            else:
                result.append(law)
        return result
