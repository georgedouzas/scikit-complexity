"""Read delayed states through hidden parameters, take events aside, and compile both with the breakpoints."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
from dataclasses import replace

import numpy as np
from sage.all__sagemath_symbolics import RR, SR, Expression
from sage.modules.free_module_element import FreeModuleElement

from ....core.laws import BaseLaw, BoundaryLaw, CrossingLaw, JumpLaw, NoiseLaw, RateLaw, UpdateLaw
from ....errors import PipelineError, ValidationError
from ....specs import EntrySpec
from ..._utils import symbol, vector
from ...problems import DeclarationsSpec, ExpressionPrinter, KernelSpec, ReferenceSpec
from ...problems._delayed_state_spec import _DelayedStateSpec
from ...types import ReadKind
from .._base import BasePass
from .._compilation import Compilation
from ._layout import LayoutPass

RELATIONS = (operator.lt, operator.le, operator.gt, operator.ge)
_SWITCHES = frozenset({'sign', 'sgn', 'heaviside', 'unit_step'})


class DelaysPass(BasePass):
    """Pass that reads delayed states through hidden parameters and takes the events of the laws aside."""

    def _get_reads(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass reads."""
        return ('ordered', 'layout')

    def _get_writes(self) -> tuple[str, ...]:
        """Return the fields of the compilation the pass writes."""
        return ('delayed', 'delays', 'placeholders', 'events')

    def _run(self, inputs: object = None) -> None:
        """Lower the delays of the ordered laws and take their events aside."""
        compilation = self.compilation_
        compilation.delays, compilation.placeholders, compilation.events = [], {}, []
        compilation.delayed = []
        for spec, laws in Compilation._required(compilation.ordered, 'ordered'):
            references: dict[Expression, ReferenceSpec] = {}
            compilation.delayed.append((spec, self._with_delays(spec, laws, compilation, references)))
            if references:
                compilation._replace_specs({id(spec): replace(spec, references={**spec.references, **references})})

    @staticmethod
    def _delay_minimum(
        spec: KernelSpec,
        tau: Expression,
        minimum: Expression | None,
        declarations: DeclarationsSpec,
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
            printer = ExpressionPrinter(time=declarations.time, compiled=True).check().prepare(symbols)
            function = printer.run([SR(tau)]).code_
            delays = function(0.0, *columns) if callable(function) else None
            if not isinstance(delays, list):
                error_msg = f'The printer returns no function of the delay of {spec.owner}.'
                raise TypeError(error_msg)
            return float(np.min(delays[0]))
        error_msg = (
            f'A delay of {spec.owner} is the expression {tau}, whose smallest value is unknown. Give it as the third '
            'argument, `delayed(x, tau, minimum)`.'
        )
        raise ValidationError(error_msg)

    @staticmethod
    def _substitute_delays(
        value: object,
        spec: KernelSpec,
        compilation: Compilation,
        references: dict[Expression, ReferenceSpec],
    ) -> object:
        """Return a law value with every delayed state replaced by a hidden parameter that solvers fill from history.

        The reference of every hidden parameter is added to `references`, which the spec receives after its laws.
        A component, or one element of its element sets, that delays one state by one delay twice reads one hidden
        parameter.
        """
        if isinstance(value, FreeModuleElement):
            return vector(DelaysPass._substitute_delays(item, spec, compilation, references) for item in value)
        if not isinstance(value, Expression):
            return value
        placeholders, delays = compilation.placeholders, compilation.delays
        if placeholders is None or delays is None:
            error_msg = 'DelaysPass reads the delays it collects, which it has not started.'
            raise PipelineError(error_msg)
        replacements = {}
        for call in BasePass._calls_of(value, 'delayed'):
            arguments = call.operands()
            item, tau = arguments[0], arguments[1]
            reference = spec.references.get(item)
            read = {ReadKind.SINGLE, ReadKind.MEMBER}
            if reference is None or not reference.is_state or reference.source not in read:
                error_msg = f'{spec.owner} delays {item}, which is not a state of an element or a group member.'
                raise ValidationError(error_msg)
            key = (spec.owner, spec.set_index, reference.entry.name, reference.column, str(tau))
            if key in placeholders:
                placeholder = placeholders[key]
                hidden = next(delay.entry for delay in delays if delay.symbol is placeholder)
                references[placeholder] = ReferenceSpec(hidden, False, 0, reference.source)
                replacements[call] = placeholder
                continue
            declarations = Compilation._required(compilation.layout, 'layout')
            entry = reference.entry
            members = entry.members if reference.source is ReadKind.MEMBER else None
            indices = (
                entry.start + np.arange(members) * entry.columns + reference.column
                if members is not None
                else np.asarray([entry.start + reference.column])
            )
            initial = np.concatenate([np.zeros(0), *declarations.initial_states])[indices]
            index = len(delays)
            hidden = EntrySpec(
                f'__delayed_{index}',
                declarations.ends[False],
                (members,) if members is not None else (),
                RR,
                members,
            )
            values = np.asarray(initial, dtype=float)
            declarations = LayoutPass._with_entry(declarations, hidden, values, is_state=False)
            compilation.layout = declarations
            placeholder = symbol(f'_delayed_{index}')
            references[placeholder] = ReferenceSpec(hidden, False, 0, reference.source)
            delays.append(
                _DelayedStateSpec(
                    entry=hidden,
                    states=np.asarray(indices, dtype=np.intp),
                    tau=tau,
                    minimum=DelaysPass._delay_minimum(spec, tau, next(iter(arguments[2:]), None), declarations),
                    spec=spec,
                    symbol=placeholder,
                    state=item,
                ),
            )
            placeholders[key] = placeholder
            replacements[call] = placeholder
        return value.subs(replacements) if replacements else value

    @staticmethod
    def _with_delays(
        spec: KernelSpec,
        laws: list[BaseLaw],
        compilation: Compilation,
        references: dict[Expression, ReferenceSpec],
    ) -> list[BaseLaw]:
        """Return laws with delayed states read through hidden parameters, keeping events aside in the compilation.

        The references of the hidden parameters are added to `references`.
        """
        events = compilation.events
        if events is None:
            error_msg = 'DelaysPass takes the events aside, which it has not started.'
            raise PipelineError(error_msg)
        result: list[BaseLaw] = []
        for law in laws:
            if isinstance(law, CrossingLaw):
                guard = DelaysPass._substitute_delays(SR(law.guard), spec, compilation, references)
                reset = {
                    target: DelaysPass._substitute_delays(SR(value), spec, compilation, references)
                    for target, value in (law.reset or {}).items()
                }
                events.append(
                    (spec, CrossingLaw(guard, reset, law.direction, law.terminal, label=law.label)),
                )
            elif isinstance(law, JumpLaw):
                result.append(
                    JumpLaw(DelaysPass._substitute_delays(law.rate, spec, compilation, references), law.changes),
                )
            elif isinstance(law, RateLaw | NoiseLaw | UpdateLaw | BoundaryLaw):
                result.append(
                    BasePass._replaced(
                        law,
                        value=DelaysPass._substitute_delays(law.value, spec, compilation, references),
                    ),
                )
            else:
                result.append(law)
        return result
