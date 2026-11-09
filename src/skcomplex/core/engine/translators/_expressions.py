"""Write the syntax structures of PySD's abstract system as source of expressions.

The source is that of the expressions of a system module.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import operator
from collections.abc import Callable, Mapping

import numpy as np
from pysd.translators.structures.abstract_expressions import (
    ArithmeticStructure,
    CallStructure,
    DelayFixedStructure,
    DelayNStructure,
    DelayStructure,
    InitialStructure,
    InlineLookupsStructure,
    LogicStructure,
    LookupsStructure,
    ReferenceStructure,
    SmoothNStructure,
    SmoothStructure,
)

from ._system_parts_spec import _SystemPartsSpec

TIME = '(t + START)'
ZERO_DIVISOR = 1e-6
INTERPOLATED = 'interpolate'
SAGE_MODULE = 'sage.all__sagemath_symbolics'
UTILITIES_MODULE = 'skcomplex.core.components.laws.utils'
OPERATORS: Mapping[str, Callable[[float, float], float]] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '^': operator.pow,
}
RELATIONS = {'=': '==', '<>': '!=', '<': '<', '<=': '<=', '>': '>', '>=': '>='}
CONTROLS = {
    'dt': 'DT',
    'time_step': 'DT',
    'saveper': 'SAVE_EVERY',
    'initial_time': 'START',
    'final_time': '(START + UNTIL)',
}
SAGE_FUNCTIONS = {
    'exp': 'exp',
    'ln': 'log',
    'log': 'log',
    'sqrt': 'sqrt',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'arcsin': 'arcsin',
    'arccos': 'arccos',
    'arctan': 'arctan',
    'sinh': 'sinh',
    'cosh': 'cosh',
    'tanh': 'tanh',
}
UNSUPPORTED = {
    'AllocateAvailableStructure': 'ALLOCATE AVAILABLE',
    'AllocateByPriorityStructure': 'ALLOCATE BY PRIORITY',
    'DataStructure': 'a data variable',
    'ForecastStructure': 'FORECAST',
    'GameStructure': 'GAME',
    'GetConstantsStructure': 'GET XLS CONSTANTS or GET DIRECT CONSTANTS',
    'GetDataStructure': 'GET XLS DATA or GET DIRECT DATA',
    'GetLookupsStructure': 'GET XLS LOOKUPS or GET DIRECT LOOKUPS',
    'SampleIfTrueStructure': 'SAMPLE IF TRUE',
    'SubscriptsReferenceStructure': 'a subscript outside a variable',
    'TrendStructure': 'TREND',
}


class _ExpressionWriter:
    """Writer of expressions of the namespace `s` and the time `t`.

    It collects what it cannot translate.
    """

    def __init__(self, parts: _SystemPartsSpec, problems: list[str], imports: set[tuple[str, str]]) -> None:
        self.parts = parts
        self.problems = problems
        self.imports = imports

    @staticmethod
    def format_number(value: float | np.number) -> str:
        """Return the source of a number, a whole number staying whole."""
        return str(int(value)) if isinstance(value, int | np.integer) else repr(float(value))

    def add_import(self, module: str, name: str) -> str:
        """Record an import of the module, and return the imported name."""
        self.imports.add((module, name))
        return name

    def _add_problem(self, construct: str) -> str:
        """Collect a construct without a counterpart.

        It returns a source that stands for it.
        """
        self.problems.append(f'{self.variable_}: {construct}')
        return '0'

    def _get_structures(self) -> Mapping[type, Callable[..., str]]:
        """Return the method that writes every syntax structure with a counterpart."""
        return {
            ArithmeticStructure: self._write_arithmetic,
            LogicStructure: self._write_logic,
            ReferenceStructure: self._write_reference,
            CallStructure: self._write_call,
            InitialStructure: self._write_initial,
            InlineLookupsStructure: self._write_inline_lookup,
            DelayStructure: self._write_delay,
            DelayNStructure: self._write_delay,
            SmoothStructure: self._write_smooth,
            SmoothNStructure: self._write_smooth,
            DelayFixedStructure: self._write_delay_fixed,
        }

    def _get_functions(self) -> Mapping[str, Callable[..., str]]:
        """Return the method that writes every built-in function with a counterpart.

        They are given by the name PySD gives them.
        """
        return {
            'abs': self._write_abs,
            'elmcount': self._write_element_count,
            'if_then_else': self._write_if_then_else,
            'integer': self._write_integer,
            'max': self._write_max,
            'min': self._write_min,
            'modulo': self._write_modulo,
            'normal': self._write_normal,
            'power': self._write_power,
            'pulse': self._write_pulse,
            'pulse_train': self._write_pulse_train,
            'ramp': self._write_ramp,
            'random': self._write_uniform,
            'random_0_1': self._write_uniform_unit,
            'random_normal': self._write_truncated_normal,
            'random_uniform': self._write_uniform,
            'step': self._write_step,
            'sum': self._write_sum,
            'xidz': self._write_xidz,
            'xpulse': self._write_single_step_pulse,
            'xpulse_train': self._write_single_step_pulse_train,
            'xramp': self._write_endless_ramp,
            'zidz': self._write_zidz,
        }

    def _write(self, ast: object) -> str:
        """Return the source of a number or a syntax structure."""
        if isinstance(ast, int | float | np.number):
            return self.format_number(ast)
        writer = self._get_structures().get(type(ast))
        if writer is None:
            name = type(ast).__name__
            return self._add_problem(UNSUPPORTED.get(name, name))
        return writer(ast)

    def _write_operand(self, ast: object) -> str:
        """Return the source of an operand, in parentheses when it is itself an operation."""
        source = self._write(ast)
        return f'({source})' if isinstance(ast, ArithmeticStructure | LogicStructure) else source

    def _write_arithmetic(self, ast: ArithmeticStructure) -> str:
        """Return the source of arithmetic, whose operators apply from left to right."""
        operands = [self._write_operand(argument) for argument in ast.arguments]
        if ast.operators == ['negative']:
            return f'-{operands[0]}'
        source = operands[0]
        for symbol, operand in zip(ast.operators, operands[1:], strict=True):
            source = f'({source}) ** {operand}' if symbol == '^' else f'{source} {symbol} {operand}'
        return source

    def _write_logic(self, ast: LogicStructure) -> str:
        """Return the source of a relation, or of a logical combination of conditions.

        The combination is a conjunction, a disjunction, or a negation.
        """
        symbol = ast.operators[0].strip(':').lower()
        if symbol == 'not':
            where = self.add_import(UTILITIES_MODULE, 'where')
            return f'{where}({self._write(ast.arguments[0])}, 0, 1) > 0'
        if symbol in {'and', 'or'}:
            name = self.add_import(UTILITIES_MODULE, 'both' if symbol == 'and' else 'either')
            operands = [self._write(argument) for argument in ast.arguments]
            return f'{name}({", ".join(operands)})'
        first, second = (self._write_operand(argument) for argument in ast.arguments)
        return f'{first} {RELATIONS[symbol]} {second}'

    def _write_condition(self, ast: object) -> str:
        """Return the source of a condition.

        A value that is not a relation holds when it is not zero.
        """
        return self._write(ast) if isinstance(ast, LogicStructure) else f'{self._write_operand(ast)} != 0'

    def _write_variable(self, name: str) -> str:
        """Return the source that reads a variable of the system by its snake-case name."""
        if name == 'time':
            return TIME
        if name in CONTROLS:
            return CONTROLS[name]
        if name in self.parts.auxiliaries:
            return f'_{name}(s, t)'
        if name in self.parts.constants or name in self.parts.stocks:
            return f's.{name}'
        return self._add_problem(f'the unknown variable {name}')

    def _write_member_read(self, source: str, index: int) -> str:
        """Return the source of a read of one member.

        It is a reduction over the member of that index.
        """
        total = self.add_import(UTILITIES_MODULE, 'total')
        return f'{total}({source}, where=s.member == {index})'

    def _write_reference(self, ast: ReferenceStructure) -> str:
        """Return the source of a reference.

        The reference is at the reading member, over a range, or at one named member.
        """
        name = _SystemPartsSpec.get_name(ast.reference)
        source = self._write_variable(name)
        subscripts = _SystemPartsSpec.get_subscripts(ast)
        if not subscripts:
            return source
        if len(subscripts) > 1:
            return self._add_problem(f'{ast.reference} subscripted over more than one range')
        subscript = subscripts[0]
        if subscript in self.parts.members:
            return self._write_member_read(source, self.parts.members.index(subscript))
        if subscript == self.parts.range_name and not self.reducing_:
            return source
        if subscript == f'{self.parts.range_name}!' and self.reducing_:
            return source
        return self._add_problem(f'{ast.reference} read over {subscript}')

    def _write_call(self, ast: CallStructure) -> str:
        """Return the source of a call of a lookup or of a built-in function."""
        name = _SystemPartsSpec.get_name(ast.function.reference)
        if name in self.parts.lookups:
            return f'_{name}({self._write(ast.arguments[0])})'
        if name in SAGE_FUNCTIONS:
            function = self.add_import(SAGE_MODULE, SAGE_FUNCTIONS[name])
            return f'{function}({", ".join(self._write(argument) for argument in ast.arguments)})'
        writer = self._get_functions().get(name)
        if writer is None:
            return self._add_problem(name.replace('_', ' ').upper())
        return writer(*ast.arguments)

    def _write_initial(self, ast: InitialStructure) -> str:
        """Return the value of an expression at the start of a run, as a number."""
        value = self.evaluate(ast.initial, 0)
        return self._add_problem('INITIAL of a value not known before a run') if value is None else repr(value)

    def write_points(self, ast: LookupsStructure) -> str:
        """Return the source of the points and the values of a table, as two tuples."""
        points = ', '.join(self.format_number(x) for x in ast.x)
        values = ', '.join(self.format_number(y) for y in ast.y)
        trailing = ',' if len(ast.x) == 1 else ''
        return f'({points}{trailing}), ({values}{trailing})'

    def _write_inline_lookup(self, ast: InlineLookupsStructure) -> str:
        """Return the source of a table written inside its variable."""
        if ast.lookups.type != INTERPOLATED:
            return self._add_problem(f'a lookup of the type {ast.lookups.type}')
        series = self.add_import(UTILITIES_MODULE, 'series')
        return f'{series}({self._write(ast.argument)}, {self.write_points(ast.lookups)})'

    def _has_equal_values(self, first: object, second: object) -> bool:
        """Check that two expressions are one expression, or have the same initial value.

        The initial value is the value at the start of a run.
        """
        if first == second:
            return True
        values = self.evaluate(first, 0), self.evaluate(second, 0)
        return None not in values and values[0] == values[1]

    def _write_delay(self, ast: DelayStructure | DelayNStructure) -> str:
        """Return the source of a material delay.

        It starts in equilibrium with its input.
        """
        if not self._has_equal_values(ast.initial, ast.input):
            return self._add_problem('a delay whose initial value differs from its input')
        arguments = f'{self._write(ast.input)}, {self._write(ast.delay_time)}'
        order = self.evaluate(ast.order, 0)
        if order is None:
            return self._add_problem('a delay of an order not known before a run')
        return f'{self.add_import(UTILITIES_MODULE, "distributed_delay")}({arguments}, {int(order)})'

    def _write_smooth(self, ast: SmoothStructure | SmoothNStructure) -> str:
        """Return the source of a smoothing.

        It starts at its input.
        """
        if not self._has_equal_values(ast.initial, ast.input):
            return self._add_problem('a smoothing whose initial value differs from its input')
        arguments = f'{self._write(ast.input)}, {self._write(ast.smooth_time)}'
        order = self.evaluate(ast.order, 0)
        if order is None:
            return self._add_problem('a smoothing of an order not known before a run')
        delay = self.add_import(UTILITIES_MODULE, 'distributed_delay')
        return f'{delay}({arguments}, {int(order)}, conserved=False)'

    def _write_delay_fixed(self, ast: DelayFixedStructure) -> str:
        """Return the source of a fixed delay of a stock.

        It reads the first value of the stock before the start.
        """
        if _SystemPartsSpec.get_reference(ast.input) not in self.parts.stocks:
            return self._add_problem('DELAY FIXED of something other than a stock')
        if not self._has_equal_values(ast.initial, ast.input):
            return self._add_problem('a fixed delay whose initial value differs from its input')
        delayed = self.add_import(UTILITIES_MODULE, 'delayed')
        return f'{delayed}({self._write(ast.input)}, {self._write(ast.delay_time)})'

    def _write_abs(self, value: object) -> str:
        """Return the source of an absolute value."""
        return f'abs({self._write(value)})'

    def _write_element_count(self, value: object) -> str:
        """Return the number of members of a range."""
        name = _SystemPartsSpec.get_reference(value)
        ranges = {_SystemPartsSpec.get_name(key): members for key, members in self.parts.ranges.items()}
        return str(len(ranges[name])) if name in ranges else self._add_problem('ELMCOUNT of an unknown range')

    def _write_if_then_else(self, condition: object, then: object, otherwise: object) -> str:
        """Return the source of a conditional."""
        where = self.add_import(UTILITIES_MODULE, 'where')
        return f'{where}({self._write_condition(condition)}, {self._write(then)}, {self._write(otherwise)})'

    def _write_integer(self, value: object) -> str:
        """Return the source of a value without its fractional part."""
        where = self.add_import(UTILITIES_MODULE, 'where')
        floor, ceil = self.add_import(SAGE_MODULE, 'floor'), self.add_import(SAGE_MODULE, 'ceil')
        source = self._write(value)
        return f'{where}({self._write_operand(value)} >= 0, {floor}({source}), {ceil}({source}))'

    @staticmethod
    def _get_bounded(inner: object, outer: object, function: str) -> tuple[object, float, float] | None:
        """Return the value and the bounds of a minimum of a maximum, or the reverse.

        The bounds are ordered numbers.
        """
        if not isinstance(inner, CallStructure) or _SystemPartsSpec.get_name(inner.function.reference) != function:
            return None
        value, bound = inner.arguments
        if not isinstance(outer, int | float | np.number) or not isinstance(bound, int | float | np.number):
            return None
        low, high = (float(bound), float(outer)) if function == 'max' else (float(outer), float(bound))
        return (value, low, high) if low <= high else None

    def _write_extreme(self, name: str, inner_name: str, first: object, second: object) -> str:
        """Return the source of a minimum or a maximum.

        When it nests the other one, it is a limit between ordered bounds.
        """
        bounded = self._get_bounded(first, second, inner_name) or self._get_bounded(second, first, inner_name)
        if bounded is None:
            function = self.add_import(UTILITIES_MODULE, name)
            return f'{function}({self._write(first)}, {self._write(second)})'
        value, low, high = bounded
        clip = self.add_import(UTILITIES_MODULE, 'clip')
        return f'{clip}({self._write(value)}, {self.format_number(low)}, {self.format_number(high)})'

    def _write_min(self, first: object, second: object) -> str:
        """Return the source of a minimum, or of a limit for a minimum of a maximum."""
        return self._write_extreme('minimum', 'max', first, second)

    def _write_max(self, first: object, second: object) -> str:
        """Return the source of a maximum, or of a limit for a maximum of a minimum."""
        return self._write_extreme('maximum', 'min', first, second)

    def _write_modulo(self, value: object, divisor: object) -> str:
        """Return the source of a remainder."""
        return f'{self.add_import(UTILITIES_MODULE, "mod")}({self._write(value)}, {self._write(divisor)})'

    def _write_power(self, base: object, exponent: object) -> str:
        """Return the source of a power."""
        return f'{self._write_operand(base)} ** {self._write_operand(exponent)}'

    def _write_pulse(self, start: object, width: object) -> str:
        """Return the source of a pulse of height one."""
        return f'{self.add_import(UTILITIES_MODULE, "pulse")}({TIME}, {self._write(start)}, {self._write(width)})'

    def _write_single_step_pulse(self, start: object, magnitude: object) -> str:
        """Return the source of a pulse of one step that moves a magnitude."""
        pulse = self.add_import(UTILITIES_MODULE, 'pulse')
        return f'{pulse}({TIME}, {self._write(start)}, DT, {self._write_operand(magnitude)} / DT)'

    def _write_pulse_train(self, start: object, interval: object, width: object, end: object) -> str:
        """Return the source of a train of pulses of height one that stops at an end time."""
        train = self.add_import(UTILITIES_MODULE, 'pulse_train')
        where = self.add_import(UTILITIES_MODULE, 'where')
        pulses = f'{train}({TIME}, {self._write(interval)}, {self._write(width)}, 1.0, {self._write(start)})'
        return f'{where}({TIME} < {self._write_operand(end)}, {pulses}, 0)'

    def _write_single_step_pulse_train(self, start: object, interval: object, magnitude: object) -> str:
        """Return the source of a train of pulses of one step that move a magnitude."""
        train = self.add_import(UTILITIES_MODULE, 'pulse_train')
        height = f'{self._write_operand(magnitude)} / DT'
        return f'{train}({TIME}, {self._write(interval)}, DT, {height}, {self._write(start)})'

    def _write_ramp(self, slope: object, start: object, end: object) -> str:
        """Return the source of a ramp of a slope between two times."""
        ramp = self.add_import(UTILITIES_MODULE, 'ramp')
        low, high = self._write_operand(start), self._write_operand(end)
        height = f'{self._write_operand(slope)} * ({high} - {low})'
        return f'{ramp}({TIME}, {self._write(start)}, {self._write(end)}, {height})'

    def _write_endless_ramp(self, slope: object, start: object) -> str:
        """Return the source of a ramp of a slope that never ends."""
        where = self.add_import(UTILITIES_MODULE, 'where')
        low = self._write_operand(start)
        return f'{where}({TIME} < {low}, 0, {self._write_operand(slope)} * ({TIME} - {low}))'

    def _write_step(self, height: object, at: object) -> str:
        """Return the source of a step of a height at a time."""
        return f'{self.add_import(UTILITIES_MODULE, "step")}({TIME}, {self._write(at)}, {self._write(height)})'

    def _write_sum(self, value: object) -> str:
        """Return the source of a sum over the members of the group."""
        if not self.parts.members:
            return self._add_problem('SUM in a system without subscripted stocks')
        self.reducing_ = True
        source = self._write(value)
        self.reducing_ = False
        return f'{self.add_import(UTILITIES_MODULE, "total")}({source})'

    def _write_zidz(self, numerator: object, denominator: object) -> str:
        """Return the source of a division that gives zero for a zero denominator."""
        return self._write_guarded_division(numerator, denominator, 0)

    def _write_xidz(self, numerator: object, denominator: object, fallback: object) -> str:
        """Return the source of a division that gives a fallback for a zero denominator."""
        return self._write_guarded_division(numerator, denominator, fallback)

    def _write_guarded_division(self, numerator: object, denominator: object, fallback: object) -> str:
        """Return the source of a division guarded against a too small denominator."""
        where = self.add_import(UTILITIES_MODULE, 'where')
        below, divisor = self._write(denominator), self._write_operand(denominator)
        division = f'{self._write_operand(numerator)} / {divisor}'
        return f'{where}(abs({below}) < {ZERO_DIVISOR}, {self._write(fallback)}, {division})'

    def _get_draw_name(self) -> str:
        """Return the name of a new draw of the variable."""
        self.draws_ += 1
        return f'{self.variable_}_{self.draws_}'

    def _write_uniform(self, low: object, high: object, *seed: object) -> str:
        """Return the source of a uniform draw."""
        del seed
        uniform = self.add_import(UTILITIES_MODULE, 'uniform')
        return f'{uniform}({self._get_draw_name()!r}, {self._write(low)}, {self._write(high)})'

    def _write_uniform_unit(self) -> str:
        """Return the source of a uniform draw between zero and one."""
        return f'{self.add_import(UTILITIES_MODULE, "uniform")}({self._get_draw_name()!r}, 0, 1)'

    def _write_normal(self, mean: object, deviation: object, *seed: object) -> str:
        """Return the source of a normal draw."""
        del seed
        normal = self.add_import(UTILITIES_MODULE, 'normal')
        return f'{normal}({self._get_draw_name()!r}, {self._write(mean)}, {self._write(deviation)})'

    def _write_truncated_normal(self, low: object, high: object, mean: object, deviation: object, *seed: object) -> str:
        """Return the source of a normal draw limited to an interval."""
        del seed
        clip = self.add_import(UTILITIES_MODULE, 'clip')
        return f'{clip}({self._write_normal(mean, deviation)}, {self._write(low)}, {self._write(high)})'

    def _evaluate_reference(self, ast: ReferenceStructure, member: int) -> float | None:
        """Return the value of a reference at the start of a run for one member.

        It returns `None` when it has none yet.
        """
        name = _SystemPartsSpec.get_name(ast.reference)
        subscripts = _SystemPartsSpec.get_subscripts(ast)
        if subscripts and subscripts[0] in self.parts.members:
            member = self.parts.members.index(subscripts[0])
        if name == 'time':
            return self.evaluate(self.parts.controls['initial_time'], member)
        if name in self.parts.controls:
            return self.evaluate(self.parts.controls[name], member)
        if name in self.parts.constants:
            values = self.parts.constants[name]
            return values[member if len(values) > 1 else 0]
        if name in self.parts.stocks:
            return self.evaluate(self.parts.stocks[name].initial, member)
        if name in self.parts.auxiliaries:
            components = self.parts.auxiliaries[name]
            return self.evaluate(components[0].ast, member) if len(components) == 1 else None
        return None

    def _evaluate_arithmetic(self, ast: ArithmeticStructure, member: int) -> float | None:
        """Return the value of arithmetic at the start of a run for one member.

        It returns `None` when it has none yet.
        """
        values = []
        for argument in ast.arguments:
            value = self.evaluate(argument, member)
            if value is None:
                return None
            values.append(value)
        if ast.operators == ['negative']:
            return -values[0]
        result = values[0]
        for symbol, value in zip(ast.operators, values[1:], strict=True):
            result = OPERATORS[symbol](result, value)
        return result

    def evaluate(self, ast: object, member: int) -> float | None:
        """Return the value of an expression at the start of a run for one member.

        It returns `None` when it has none yet.
        """
        if isinstance(ast, int | float | np.number):
            return float(ast)
        if isinstance(ast, InitialStructure):
            return self.evaluate(ast.initial, member)
        if isinstance(ast, ReferenceStructure):
            return self._evaluate_reference(ast, member)
        return self._evaluate_arithmetic(ast, member) if isinstance(ast, ArithmeticStructure) else None

    def write(self, ast: object, variable: str) -> str:
        """Return the source of an expression of a variable.

        It collects what it cannot translate under that name.
        """
        self.variable_ = variable
        self.reducing_ = False
        self.draws_ = 0
        return self._write(ast)
