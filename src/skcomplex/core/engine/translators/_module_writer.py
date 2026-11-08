"""Write the source of a system module from the sorted parts of a stock-and-flow system."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
import textwrap
from collections.abc import Mapping, Sequence

from pysd.translators.structures.abstract_expressions import ArithmeticStructure

from ...specs import RunSettingsSpec
from ._expressions import SAGE_MODULE, UTILITIES_MODULE, _ExpressionWriter
from ._system_parts_spec import _GoalSeekingSpec, _SystemPartsSpec

LINE_LENGTH = 120
INDENT = '    '
ELEMENTS_MODULE = 'skcomplex.core.components.elements'
INTERACTIONS_MODULE = 'skcomplex.core.components.interactions'
LAWS_MODULE = 'skcomplex.core.components.laws'
SPECS_MODULE = 'skcomplex.core.specs'
BASE_IMPORTS = frozenset(
    {
        (SAGE_MODULE, 'Expression'),
        (ELEMENTS_MODULE, 'Element'),
        ('skcomplex.core.components.system', 'BaseSystem'),
        ('skcomplex.core.components.system', 'System'),
        (SPECS_MODULE, 'ObservableSpec'),
        (SPECS_MODULE, 'StateSpec'),
        (SPECS_MODULE, 'VisualSpec'),
    },
)
ELEMENT_CLASS = '''class {name}(Element):
    """Hold the stocks and the constants of the system.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

{arguments}
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
{declarations}
    ) -> None:
        """Store the label, the visual, the stocks, and the constants."""
        super().__init__(label, visual=visual)
{assignments}'''
TRANSFER = """                TransferInteraction(
                    {name!r},
                    source={label!r},
                    target={label!r},
                    state={state},
                    into={into},
                    amount=_amount__{name},
                    flow={name!r},
                ),"""
RELAXATION = """                RelaxationInteraction(
                    {name!r},
                    target={label!r},
                    state={stock!r},
                    goal=_goal__{stock},
                    time=_time__{stock},
                ),"""
RATE = """                RateInteraction(
                    {name!r},
                    target={label!r},
                    state={stock!r},
                    rate=_rate__{stock},
                ),"""
GROUP = """                GroupElement(
                    {label!r},
                    element={name},
                    size={size},
                    values={{
{values}
                    }},
                ),"""
SYSTEM_CLASS = '''class {name}(BaseSystem):
    """Run the stock-and-flow system translated from the file `{stem}`.

    Args:
{arguments}
    """

    until = UNTIL
    dt = SAVE_EVERY
    observables = ({observables})

    def __init__(
        self,
        *,
{declarations}
    ) -> None:
        """Store the constants and the label."""
        super().__init__(label)
{assignments}

    def _build_system(self) -> System:
        """Return the element of the stocks and the constants, and the interactions of the flows and the rates."""
        return System(
            self.label,
            elements=[
{element}
            ],
            interactions=[
{interactions}
            ],
        )'''


class _ModuleWriter:
    """Writer of the source of the module of a system, one block of source per part."""

    def __init__(
        self,
        parts: _SystemPartsSpec,
        settings: RunSettingsSpec,
        label: str,
        stem: str,
        expressions: _ExpressionWriter,
    ) -> None:
        self.parts = parts
        self.settings = settings
        self.label = label
        self.stem = stem
        self.expressions = expressions

    @staticmethod
    def _write_return(source: str) -> str:
        """Return the return statement of a function body, on several lines when it does not fit on one."""
        line = f'{INDENT}return {source}'
        if len(line) <= LINE_LENGTH:
            return line
        lines = textwrap.wrap(
            source,
            width=LINE_LENGTH - 2 * len(INDENT),
            break_long_words=False,
            break_on_hyphens=False,
        )
        body = '\n'.join(f'{INDENT * 2}{item}' for item in lines)
        return f'{INDENT}return (\n{body}\n{INDENT})'

    @staticmethod
    def _write_function(signature: str, summary: str, *body: str) -> str:
        """Return the source of a function with one line of documentation."""
        return '\n'.join([f'def {signature}:', f'{INDENT}"""{summary}"""', *body])

    @staticmethod
    def _write_bindings(source: str) -> list[str]:
        """Return the lines that bind the namespace of the target and the time, for those a source reads."""
        lines = [f'{INDENT}s = namespace.target'] if re.search(r'\bs\b', source) else []
        return [*lines, f'{INDENT}t = namespace.t'] if re.search(r'\bt\b', source) else lines

    def _get_original(self, name: str) -> str:
        """Return a name of the file without what a docstring of the module cannot hold."""
        return self.parts.originals[name].replace('\\', '').replace('"', '').strip()

    def _get_label(self, base: str) -> str:
        """Return a label of a generated interaction that no variable of the file holds."""
        label = base
        while label in self.parts.originals:
            label = f'{label}_generated'
        return label

    def _get_class_name(self) -> str:
        """Return the name of the element class, after the label of the element."""
        return ''.join(word.capitalize() for word in self.label.split('_')) + 'Element'

    def _get_stock_values(self, name: str) -> list[float]:
        """Return the first value of a stock for every member, and collect one not known before a run."""
        parts = self.parts
        values = [
            self.expressions.evaluate(parts.stocks[name].initial, member) for member in range(len(parts.members) or 1)
        ]
        if None in values:
            self.expressions.problems.append(f'{parts.originals[name]}: an initial value not known before a run')
        return [0.0 if value is None else value for value in values]

    def _write_choice(self, name: str) -> str:
        """Return the source of an auxiliary, a choice by the member when a component defines one member alone."""
        parts, expressions = self.parts, self.expressions
        components = parts.auxiliaries[name]
        if len(components) == 1:
            return expressions.write(components[0].ast, parts.originals[name])
        where = expressions.add_import(UTILITIES_MODULE, 'where')
        source = '0'
        for component in reversed(components):
            value = expressions.write(component.ast, parts.originals[name])
            for member in parts.get_covered(component) or []:
                source = f'{where}(s.member == {member}, {value}, {source})'
        return source

    def _write_helpers(self) -> list[str]:
        """Return the source of the function of every lookup and every auxiliary."""
        parts, expressions = self.parts, self.expressions
        helpers = []
        for name, table in sorted(parts.lookups.items()):
            lookup = expressions.add_import(UTILITIES_MODULE, 'lookup')
            helpers.append(
                self._write_function(
                    f'_{name}(x: Expression) -> Expression',
                    f'Return the lookup `{self._get_original(name)}`.',
                    self._write_return(f'{lookup}(x, {expressions.write_points(table)})'),
                ),
            )
        for name in parts.order_auxiliaries(expressions.problems):
            helpers.append(
                self._write_function(
                    f'_{name}(s: SimpleNamespace, t: Expression) -> Expression',
                    f'Return the auxiliary `{self._get_original(name)}`.',
                    self._write_return(self._write_choice(name)),
                ),
            )
        return helpers

    def _write_element(self) -> str:
        """Return the source of the element class, whose states are the stocks and whose parameters the constants."""
        parts = self.parts
        declared = [(name, 'StateSpec', self._get_stock_values(name)[0], 'stock') for name in sorted(parts.stocks)]
        declared += [(name, 'ParameterSpec', parts.constants[name][0], 'constant') for name in sorted(parts.constants)]
        documented = [
            f'{INDENT * 2}{name}:\n{INDENT * 3}The {kind} `{self._get_original(name)}`.'
            for name, _, _, kind in declared
        ]
        if parts.constants or parts.members:
            self.expressions.imports.add((SPECS_MODULE, 'ParameterSpec'))
        if parts.members:
            declared.append(('member', 'ParameterSpec', 0.0, 'position'))
            documented.append(f'{INDENT * 2}member:\n{INDENT * 3}The position of the member in the range of the file.')
        return ELEMENT_CLASS.format(
            name=self._get_class_name(),
            arguments='\n\n'.join(documented),
            declarations='\n'.join(
                f'{INDENT * 2}{name}: Annotated[float, {spec}()] = {value!r},' for name, spec, value, _ in declared
            ),
            assignments='\n'.join(f'{INDENT * 2}self.{name} = {name}' for name, _, _, _ in declared),
        )

    def _write_transfers(self, flows: dict[str, tuple[str | None, str | None]]) -> tuple[list[str], list[str]]:
        """Return the functions of the amounts of the flows, and the source of their transfers."""
        imports = self.expressions.imports
        functions, interactions = [], []
        for name, (leaving, entering) in flows.items():
            functions.append(
                self._write_function(
                    f'_amount__{name}(s: SimpleNamespace) -> Expression',
                    f'Return the amount of the flow `{self._get_original(name)}`.',
                    f'{INDENT}return _{name}(s.source, s.t)',
                ),
            )
            if leaving is None or entering is None:
                imports.add((LAWS_MODULE, 'Source' if leaving is None else 'Sink'))
            state = 'Source()' if leaving is None else repr(leaving)
            into = 'Sink()' if entering is None else repr(entering)
            interactions.append(TRANSFER.format(name=name, label=self.label, state=state, into=into))
            imports.add((INTERACTIONS_MODULE, 'TransferInteraction'))
        return functions, interactions

    def _write_relaxation(self, stock: str, form: _GoalSeekingSpec) -> tuple[list[str], str, str]:
        """Return the functions of a goal-seeking stock, the source of its relaxation, and the label of its rate."""
        expressions = self.expressions
        original = self.parts.originals[stock]
        expressions.imports.add((INTERACTIONS_MODULE, 'RelaxationInteraction'))
        goal = expressions.write(form.goal, original)
        scale = expressions.write(form.scale, original)
        time = scale if form.is_time else f'1 / ({scale})'
        name = self._get_label(f'{stock}_adjustment')
        functions = [
            self._write_function(
                f'_goal__{stock}(namespace: SimpleNamespace) -> Expression',
                f'Return the goal of the stock `{self._get_original(stock)}`.',
                *self._write_bindings(goal),
                self._write_return(goal),
            ),
            self._write_function(
                f'_time__{stock}(namespace: SimpleNamespace) -> Expression',
                f'Return the adjustment time of the stock `{self._get_original(stock)}`.',
                *self._write_bindings(time),
                self._write_return(time),
            ),
            self._write_function(
                f'_observed__{name}(s: SimpleNamespace) -> Expression',
                f'Return the rate that brings the stock `{self._get_original(stock)}` to its goal.',
                *([f'{INDENT}t = s.t'] if re.search(r'\bt\b', f'{goal} {time}') else []),
                self._write_return(f'(({goal}) - s.{stock}) / ({time})'),
            ),
        ]
        return functions, RELAXATION.format(name=name, label=self.label, stock=stock), name

    def _write_rate(self, stock: str, terms: Sequence[tuple[int, object]]) -> tuple[str, str]:
        """Return the function of what remains of the net rate of a stock, and the source of its rate interaction."""
        expressions = self.expressions
        expressions.imports.add((INTERACTIONS_MODULE, 'RateInteraction'))
        pieces = []
        for sign, term in terms:
            source = expressions.write(term, self.parts.originals[stock])
            operand = f'({source})' if isinstance(term, ArithmeticStructure) else source
            pieces.append(f'{"+" if sign > 0 else "-"} {operand}')
        net = ' '.join(pieces).removeprefix('+ ')
        function = self._write_function(
            f'_rate__{stock}(namespace: SimpleNamespace) -> Expression',
            f'Return what remains of the net rate of the stock `{self._get_original(stock)}`.',
            *self._write_bindings(net),
            self._write_return(net),
        )
        return function, RATE.format(name=self._get_label(f'{stock}_rate'), label=self.label, stock=stock)

    def _write_observable(self, name: str) -> str:
        """Return the source of the observable of a name, of the element or reduced over the group."""
        if not self.parts.members:
            return f'{INDENT * 2}ObservableSpec({name!r}, _observed__{name}, element={self.label!r}),'
        reduction = '_member_values' if self.parts.is_per_member(name) else "'mean'"
        return f'{INDENT * 2}ObservableSpec({name!r}, _observed__{name}, group={self.label!r}, reduce={reduction}),'

    def _write_observables(self, flows: Mapping[str, object], relaxations: Sequence[str]) -> tuple[list[str], str]:
        """Return the functions of the observables, and the source of the items of their tuple."""
        parts = self.parts
        names = [name for name in sorted(parts.auxiliaries) if name not in flows and parts.is_observable(name)]
        functions = [
            self._write_function(
                f'_observed__{name}(s: SimpleNamespace) -> Expression',
                f'Return the auxiliary `{self._get_original(name)}` at a frame.',
                f'{INDENT}return _{name}(s, s.t)',
            )
            for name in names
        ]
        if parts.members:
            functions.append(
                self._write_function(
                    '_member_values(values: object) -> object',
                    'Return the values of the members of the group, one row per member.',
                    f'{INDENT}return values',
                ),
            )
        specs = '\n'.join(self._write_observable(name) for name in [*names, *relaxations])
        return functions, f'\n{specs}\n{INDENT}' if specs else ''

    def _get_system_name(self) -> str:
        """Return the name of the system class, after the label of the element."""
        return self._get_class_name().removesuffix('Element') + 'System'

    def _write_element_call(self) -> str:
        """Return the source of the element of the system, given the constants of the system."""
        parts = self.parts
        name = self._get_class_name()
        if parts.members:
            self.expressions.imports.add((ELEMENTS_MODULE, 'GroupElement'))
            values: dict[str, str] = {stock: repr(self._get_stock_values(stock)) for stock in sorted(parts.stocks)}
            for key, value in sorted(parts.constants.items()):
                values[key] = f'list(self.{key})' if len(value) > 1 else f'self.{key}'
            values['member'] = repr([float(index) for index in range(len(parts.members))])
            listed = '\n'.join(f'{INDENT * 6}{key!r}: {value},' for key, value in values.items())
            return GROUP.format(label=self.label, name=name, size=len(parts.members), values=listed)
        if not parts.constants:
            return f'{INDENT * 4}{name}({self.label!r}),'
        arguments = '\n'.join(f'{INDENT * 5}{key}=self.{key},' for key in sorted(parts.constants))
        return f'{INDENT * 4}{name}(\n{INDENT * 5}{self.label!r},\n{arguments}\n{INDENT * 4}),'

    def _write_system(self, interactions: Sequence[str], observables: str) -> str:
        """Return the source of the system class, whose parameters are the constants of the file."""
        parts = self.parts
        declared = []
        for key, value in sorted(parts.constants.items()):
            declared.append((key, f'tuple[float, ...] = {value!r}' if len(value) > 1 else f'float = {value[0]!r}'))
        documented = [
            f'{INDENT * 2}{key}:\n{INDENT * 3}The constant `{self._get_original(key)}`.' for key, _ in declared
        ]
        label = 'The label of the system, which names it when it is nested in another one.'
        documented.append(f'{INDENT * 2}label:\n{INDENT * 3}{label}')
        declarations = [f'{INDENT * 2}{key}: {annotation},' for key, annotation in declared]
        declarations.append(f"{INDENT * 2}label: str = 'system',")
        return SYSTEM_CLASS.format(
            name=self._get_system_name(),
            stem=self.stem,
            arguments='\n\n'.join(documented),
            observables=observables,
            declarations='\n'.join(declarations),
            assignments='\n'.join(f'{INDENT * 2}self.{key} = {key}' for key, _ in declared),
            element=self._write_element_call(),
            interactions='\n'.join(interactions),
        )

    @staticmethod
    def _write_import(module: str, names: Sequence[str]) -> str:
        """Return the import of names of a module, on several lines when it does not fit on one."""
        line = f'from {module} import {", ".join(names)}'
        if len(line) <= LINE_LENGTH:
            return line
        listed = '\n'.join(f'{INDENT}{name},' for name in names)
        return f'from {module} import (\n{listed}\n)'

    def _write_imports(self) -> str:
        """Return the import block of the module, the standard library first and the library last."""
        modules: dict[str, list[str]] = {}
        for module, name in self.expressions.imports:
            modules.setdefault(module, []).append(name)
        sage = ', '.join(sorted(modules.pop(SAGE_MODULE)))
        library = '\n'.join(self._write_import(module, sorted(names)) for module, names in sorted(modules.items()))
        return (
            'from types import SimpleNamespace\nfrom typing import Annotated\n\n'
            f'from sage.all__sagemath_symbolics import {sage}\n\n{library}'
        )

    def write(self) -> str:
        """Return the source of the module of the system."""
        self.expressions.imports |= BASE_IMPORTS
        parts, settings = self.parts, self.settings
        flows = parts.get_flows()
        helpers = self._write_helpers()
        amounts, interactions = self._write_transfers(flows)
        rates, relaxations = [], []
        for stock in sorted(parts.stocks):
            terms = parts.get_remaining_terms(stock, flows)
            form = parts.get_goal_seeking(stock, terms)
            if form is not None:
                functions, interaction, name = self._write_relaxation(stock, form)
                rates += functions
                relaxations.append(name)
                interactions.append(interaction)
            elif terms:
                function, interaction = self._write_rate(stock, terms)
                rates.append(function)
                interactions.append(interaction)
        observed, observables = self._write_observables(flows, relaxations)
        blocks = [
            *helpers,
            self._write_element(),
            *amounts,
            *rates,
            *observed,
            self._write_system(interactions, observables),
        ]
        constants = (
            f'START = {settings.start!r}\nUNTIL = {settings.until!r}\n'
            f'DT = {settings.dt!r}\nSAVE_EVERY = {settings.save_every!r}'
        )
        header = f'"""Define the stock-and-flow system translated from the file `{self.stem}`."""'
        return f'{header}\n\n{self._write_imports()}\n\n{constants}\n\n\n' + '\n\n\n'.join(blocks) + '\n'
