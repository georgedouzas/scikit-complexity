"""Translate Vensim and XMILE files into the source of system modules.

They are translated through the abstract system of PySD.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import keyword
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from pysd.translators.structures.abstract_expressions import IntegStructure, LookupsStructure
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractControlElement,
    AbstractElement,
    AbstractLookup,
    AbstractModel,
    AbstractSection,
)
from pysd.translators.vensim.vensim_file import VensimFile
from pysd.translators.xmile.xmile_file import XmileFile

from ...base import BaseConfig
from ...components.elements import Element
from ...components.systems import BaseSystem
from ...errors import ValidationError
from ...specs import RunSettingsSpec
from ._expressions import INTERPOLATED, _ExpressionWriter
from ._module_writer import _ModuleWriter
from ._system_parts_spec import _SystemPartsSpec

VENSIM_SUFFIX = '.mdl'
XMILE_SUFFIXES = ('.xmile', '.xml', '.stmx')
MAIN_SECTION = 'main'
CONTROL_NAMES = ('initial_time', 'final_time', 'time_step', 'saveper')
RESERVED_NAMES = frozenset(
    {'t', 'time', 'member', 'label', 'visual', *CONTROL_NAMES, *dir(Element), *dir(BaseSystem)},
)


class SystemDynamicsTranslator(BaseConfig[None, None]):
    """Translator of a Vensim or XMILE file into the source of a system module.

    The module is a system module of the library. The module holds one element class, whose states are the stocks and
    whose parameters are the constants of the file, one private function per auxiliary and lookup, the constants
    `START`, `UNTIL`, `DT`, and `SAVE_EVERY`, and one system class on `BaseSystem`, named after the element in CamelCase
    with the suffix `System`. The system class takes the constants as keyword arguments, with the values of the file as
    defaults, builds its element and its interactions in `_build_system`, and holds as tuned run settings `until`, the
    length of the run, `dt`, the saving interval of the file, and `observables`, one per auxiliary and per goal-seeking
    stock. A flow variable of two stocks becomes a `TransferInteraction` between them, a flow variable of one stock a
    `TransferInteraction` with `Source()` or `Sink()`, and what remains of the rate of a stock a `RelaxationInteraction`
    or a `RateInteraction`. A system whose stocks are all subscripted over one range becomes a `GroupElement` with one
    member per subscript.

    Args:
        path:
            The path of a `.mdl`, `.xmile`, `.xml`, or `.stmx` file.

        name:
            The name of the element and the base of the name of its class, or `None` for the name of the file.

    Attributes:
        system_:
            The abstract system PySD reads from the file.

        source_:
            The Python source of the system module.

        settings_:
            The run settings of the control section of the file.

    Raises:
        ValidationError:
            The path names no file of a known format, or the name is not an identifier, raised by `prepare`. The file
            has constructs without a counterpart, or names that clash, all of them listed, raised by `run`.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from skcomplex.core.engine.translators import SystemDynamicsTranslator
        >>> text = '''
        ... Stock = INTEG(Inflow, 1) ~ ~ |
        ... Inflow = 2 * Stock ~ ~ |
        ... INITIAL TIME = 0 ~ ~ |
        ... FINAL TIME = 4 ~ ~ |
        ... TIME STEP = 0.25 ~ ~ |
        ... SAVEPER = 1 ~ ~ |
        ... '''
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = Path(folder) / 'growth.mdl'
        ...     _ = path.write_text(text)
        ...     translator = SystemDynamicsTranslator(path).prepare().run()
        >>> translator.settings_.until
        4.0
        >>> 'class GrowthElement(Element):' in translator.source_
        True
        >>> 'class GrowthSystem(BaseSystem):' in translator.source_
        True
    """

    def __init__(self, path: str | Path, *, name: str | None = None) -> None:
        self.path = path
        self.name = name

    def _prepare(self, context: None) -> None:
        """Read the abstract system of the file with the translator of PySD for its format.

        It first raises an error for a path of an unknown format, or a name that is no identifier.

        Args:
            context:
                Nothing, since the translator reads its file alone.

        Raises:
            ValidationError:
                The path names no file of a known format, or the name is not an identifier.
        """
        path = Path(self.path)
        if path.suffix.lower() not in {VENSIM_SUFFIX, *XMILE_SUFFIXES}:
            error_msg = (
                f'The file {path} has an unknown format. Give a {VENSIM_SUFFIX} file or one of {XMILE_SUFFIXES}.'
            )
            raise ValidationError(error_msg)
        if not path.is_file():
            error_msg = f'The file {path} does not exist. Give the path of a system file.'
            raise ValidationError(error_msg)
        if self.name is not None and not self.name.isidentifier():
            error_msg = f'The name {self.name!r} is not an identifier. Give a snake-case name, or None.'
            raise ValidationError(error_msg)
        path = Path(self.path)
        reader = VensimFile(path) if path.suffix.lower() == VENSIM_SUFFIX else XmileFile(path)
        reader.parse()
        self.system_: AbstractModel = reader.get_abstract_model()

    @staticmethod
    def _get_ranges(section: AbstractSection) -> Mapping[str, tuple[str, ...]]:
        """Return the subscripts of every range of a section that lists them."""
        return {item.name: tuple(item.subscripts) for item in section.subscripts if isinstance(item.subscripts, list)}

    @staticmethod
    def _is_stock(element: AbstractElement) -> bool:
        """Check that a variable is a stock."""
        return any(isinstance(component.ast, IntegStructure) for component in element.components)

    def _read_range(self, elements: Sequence[AbstractElement], problems: list[str]) -> str:
        """Return the one range the stocks are subscripted over, and collect any other.

        It returns the empty string for none.
        """
        subscripts = {
            element.name: element.components[0].subscripts[0] for element in elements if self._is_stock(element)
        }
        ranges = {subscript for names in subscripts.values() for subscript in names}
        unsubscripted = [name for name, names in subscripts.items() if not names]
        if ranges and unsubscripted:
            problems += [
                f'{name}: an unsubscripted stock in a system with subscripted stocks' for name in unsubscripted
            ]
        if len(ranges) > 1:
            problems += [f'{name}: a stock subscripted over more than one range' for name in subscripts]
            return ''
        return next(iter(ranges), '')

    @staticmethod
    def _read_constant(element: AbstractElement, members: tuple[str, ...]) -> tuple[float, ...] | None:
        """Return the value or values of a variable that holds only numbers.

        There is one value per member when it has several. It returns `None` otherwise.
        """
        values: list[float] = []
        for component in element.components:
            ast = component.ast
            if isinstance(ast, np.ndarray) and ast.ndim == 1:
                values += [float(value) for value in ast]
            elif isinstance(ast, int | float | np.number):
                values.append(float(ast))
            else:
                return None
        if not element.components[0].subscripts[0]:
            return tuple(values)
        return tuple(values if len(values) == len(members) else values * len(members))

    @staticmethod
    def _read_lookup(element: AbstractElement, lookups: dict[str, LookupsStructure], problems: list[str]) -> None:
        """Keep the table of a lookup, and collect a lookup that cannot translate.

        Only a table of one kind translates.
        """
        ast = element.components[0].ast
        if len(element.components) > 1 or not isinstance(ast, LookupsStructure):
            problems.append(f'{element.name}: a lookup that is not one table of numbers')
        elif ast.type != INTERPOLATED:
            problems.append(f'{element.name}: a lookup of the type {ast.type}')
        else:
            lookups[_SystemPartsSpec.get_name(element.name)] = ast

    @staticmethod
    def _read_stock(element: AbstractElement, stocks: dict[str, IntegStructure], problems: list[str]) -> None:
        """Keep the integral of a stock, and collect a stock defined apart per subscript."""
        if len(element.components) > 1:
            problems.append(f'{element.name}: a stock defined apart for every subscript')
        else:
            stocks[_SystemPartsSpec.get_name(element.name)] = element.components[0].ast

    def _read_parts(self, problems: list[str]) -> _SystemPartsSpec:
        """Return the variables of the main section sorted by what they become.

        It collects what cannot become any.
        """
        main = next(section for section in self.system_.sections if section.type == MAIN_SECTION)
        problems += [f'{section.name}: a macro' for section in self.system_.sections if section.type != MAIN_SECTION]
        elements = [item for item in main.elements if not isinstance(item, AbstractControlElement)]
        ranges = self._get_ranges(main)
        range_name = self._read_range(elements, problems)
        members = ranges.get(range_name, ())
        if range_name and not members:
            problems.append(f'{range_name}: a subscript range that does not list its subscripts')
        constants: dict[str, tuple[float, ...]] = {}
        stocks: dict[str, IntegStructure] = {}
        auxiliaries: dict[str, Sequence[AbstractComponent]] = {}
        lookups: dict[str, LookupsStructure] = {}
        for element in elements:
            name = _SystemPartsSpec.get_name(element.name)
            constant = self._read_constant(element, members)
            if isinstance(element.components[0], AbstractLookup):
                self._read_lookup(element, lookups, problems)
            elif self._is_stock(element):
                self._read_stock(element, stocks, problems)
            elif constant is not None:
                constants[name] = constant
            else:
                auxiliaries[name] = element.components
        return _SystemPartsSpec(
            originals={_SystemPartsSpec.get_name(element.name): element.name for element in elements},
            controls={
                _SystemPartsSpec.get_name(item.name): item.components[0].ast
                for item in main.elements
                if isinstance(item, AbstractControlElement)
            },
            constants=constants,
            stocks=stocks,
            auxiliaries=auxiliaries,
            lookups=lookups,
            ranges=ranges,
            members=members,
            range_name=range_name,
        )

    def _check_subscripts(self, parts: _SystemPartsSpec, problems: list[str]) -> None:
        """Collect the variables subscripted over another range than that of the stocks.

        This includes variables subscripted over more than one range.
        """
        for section in self.system_.sections:
            for element in section.elements:
                for component in element.components:
                    subscripts = component.subscripts[0]
                    if len(subscripts) > 1:
                        problems.append(f'{element.name}: a variable subscripted over more than one range')
                    elif subscripts and parts.get_covered(component) is None:
                        problems.append(f'{element.name}: a variable subscripted over {subscripts[0]}')

    def _check_names(self, parts: _SystemPartsSpec, label: str, problems: list[str]) -> None:
        """Collect the names that collide, clash with the module, or are no identifiers."""
        seen: dict[str, list[str]] = {}
        for section in self.system_.sections:
            for element in section.elements:
                if not isinstance(element, AbstractControlElement):
                    seen.setdefault(_SystemPartsSpec.get_name(element.name), []).append(element.name)
        for name, originals in seen.items():
            if len(originals) > 1:
                problems.append(f'{" and ".join(originals)}: names that share the snake-case name {name}')
            if name in RESERVED_NAMES or not name.isidentifier() or keyword.iskeyword(name):
                problems.append(f'{originals[0]}: a name that clashes with the system module as {name}')
        if label in parts.auxiliaries:
            problems.append(f'{parts.originals[label]}: a name that clashes with the element. Give `name` to rename it')
        if not label.isidentifier() or keyword.iskeyword(label):
            problems.append(f'{self.path}: a file that names the element {label!r}. Give `name` to rename it')

    @staticmethod
    def _read_settings(parts: _SystemPartsSpec, expressions: _ExpressionWriter, problems: list[str]) -> RunSettingsSpec:
        """Return the run settings of the control section.

        Its length runs from its start.
        """
        values = {}
        for name in CONTROL_NAMES:
            value = expressions.evaluate(parts.controls[name], 0)
            if value is None:
                problems.append(f'{name}: a control value not known before a run')
            values[name] = 0.0 if value is None else value
        start = values['initial_time']
        return RunSettingsSpec(start, values['final_time'] - start, values['time_step'], values['saveper'])

    def _run(self, inputs: None) -> None:
        """Translate the abstract system.

        It sets the source of the module and the run settings.

        Args:
            inputs:
                Nothing, since the translator reads the file it prepared.

        Raises:
            ValidationError:
                The file has constructs without a counterpart, or names that clash, all of them listed.
        """
        problems: list[str] = []
        parts = self._read_parts(problems)
        stem = Path(self.path).stem
        label = self.name or _SystemPartsSpec.get_name(stem)
        self._check_subscripts(parts, problems)
        self._check_names(parts, label, problems)
        expressions = _ExpressionWriter(parts, problems, set())
        settings = self._read_settings(parts, expressions, problems)
        source = _ModuleWriter(parts, settings, label, stem, expressions).write()
        if problems:
            listed = '\n'.join(f'- {problem}' for problem in dict.fromkeys(problems))
            error_msg = f'The file {self.path} has constructs that do not translate:\n{listed}'
            raise ValidationError(error_msg)
        settings._check()
        self.settings_ = settings
        self.source_ = source
