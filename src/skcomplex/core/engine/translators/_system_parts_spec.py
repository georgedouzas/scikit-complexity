"""Define the parts of a stock-and-flow system that a file declares."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass

from pysd.translators.structures.abstract_expressions import (
    ArithmeticStructure,
    CallStructure,
    DelayFixedStructure,
    DelayNStructure,
    DelayStructure,
    IntegStructure,
    LookupsStructure,
    ReferenceStructure,
    SmoothNStructure,
    SmoothStructure,
)
from pysd.translators.structures.abstract_model import AbstractComponent

from ...specs import BaseSpec

DELAYS = (DelayStructure, DelayNStructure, SmoothStructure, SmoothNStructure)


@dataclass(frozen=True)
class _GoalSeekingSpec(BaseSpec):
    """Form `(goal - stock) / time` of a net rate.

    It is `(goal - stock) * rate` when the scale is no time.
    """

    goal: object
    scale: object
    is_time: bool


@dataclass(frozen=True)
class _SystemPartsSpec(BaseSpec):
    """Variables of a system file by snake-case name, sorted by kind.

    The kinds are stocks, constants, auxiliaries, and lookups.
    """

    originals: Mapping[str, str]
    controls: Mapping[str, object]
    constants: Mapping[str, tuple[float, ...]]
    stocks: Mapping[str, IntegStructure]
    auxiliaries: Mapping[str, Sequence[AbstractComponent]]
    lookups: Mapping[str, LookupsStructure]
    ranges: Mapping[str, tuple[str, ...]]
    members: tuple[str, ...]
    range_name: str

    @staticmethod
    def get_name(original: str) -> str:
        """Return the snake-case name of a name of a file."""
        return re.sub(r'[\W_]+', '_', original.lower()).strip('_')

    @staticmethod
    def get_reference(ast: object) -> str:
        """Return the snake-case name a plain reference reads.

        It returns the empty string for anything else.
        """
        return _SystemPartsSpec.get_name(ast.reference) if isinstance(ast, ReferenceStructure) else ''

    @staticmethod
    def get_subscripts(ast: ReferenceStructure) -> Sequence[str]:
        """Return the subscripts of a reference."""
        return ast.subscripts.subscripts if ast.subscripts is not None else ()

    @staticmethod
    def _get_children(ast: object) -> Iterator[object]:
        """Yield the expressions directly inside a syntax structure."""
        if not is_dataclass(ast):
            return
        for field in fields(ast):
            value = getattr(ast, field.name)
            if isinstance(value, list | tuple):
                yield from value
            else:
                yield value

    @staticmethod
    def _split_terms(ast: object, sign: int) -> Iterator[tuple[int, object]]:
        """Yield the signed terms of the sums and differences of a net rate."""
        if not isinstance(ast, ArithmeticStructure) or not set(ast.operators) <= {'+', '-', 'negative'}:
            yield sign, ast
            return
        if ast.operators == ['negative']:
            yield from _SystemPartsSpec._split_terms(ast.arguments[0], -sign)
            return
        signs = (1, *(1 if symbol == '+' else -1 for symbol in ast.operators))
        for term_sign, argument in zip(signs, ast.arguments, strict=True):
            yield from _SystemPartsSpec._split_terms(argument, sign * term_sign)

    def _holds(self, ast: object, is_match: Callable[[object], bool], visited: set[str]) -> bool:
        """Check that an expression holds a match, directly or through auxiliaries.

        It adds those it visits.
        """
        if is_match(ast):
            return True
        name = self.get_reference(ast)
        if name in visited:
            return False
        if name in self.auxiliaries:
            visited.add(name)
            return any(self._holds(component.ast, is_match, visited) for component in self.auxiliaries[name])
        return any(self._holds(child, is_match, visited) for child in self._get_children(ast))

    def _is_reduction(self, ast: object) -> bool:
        """Check that an expression is a sum over the group, or a read of one named member."""
        if isinstance(ast, CallStructure):
            return self.get_name(ast.function.reference) == 'sum'
        return isinstance(ast, ReferenceStructure) and any(item in self.members for item in self.get_subscripts(ast))

    def _is_flow_variable(self, term: object) -> bool:
        """Check that a term of a net rate is a plain read of a recordable auxiliary.

        The amount of the auxiliary must be one that a trajectory can record. A trajectory records the amount of one
        member from the values of that member, so it cannot read other members.
        """
        if not isinstance(term, ReferenceStructure) or self.get_name(term.reference) not in self.auxiliaries:
            return False
        if list(self.get_subscripts(term)) not in ([], [self.range_name]):
            return False
        return not self._holds(term, self._is_reduction, set())

    def get_flows(self) -> dict[str, tuple[str | None, str | None]]:
        """Return the stocks every flow variable leaves and enters.

        Only those that move between stocks count. A variable that enters or leaves more stocks than a transfer has ends
        stays in the remaining rates.
        """
        visits: dict[str, list[tuple[str, int]]] = {}
        for stock, integral in sorted(self.stocks.items()):
            for sign, term in self._split_terms(integral.flow, 1):
                if self._is_flow_variable(term):
                    visits.setdefault(self.get_reference(term), []).append((stock, sign))
        flows: dict[str, tuple[str | None, str | None]] = {}
        for name, seen in sorted(visits.items()):
            leaving = [stock for stock, sign in seen if sign < 0]
            entering = [stock for stock, sign in seen if sign > 0]
            if len(leaving) <= 1 and len(entering) <= 1 and set(leaving).isdisjoint(entering):
                flows[name] = (next(iter(leaving), None), next(iter(entering), None))
        return flows

    def get_remaining_terms(self, stock: str, flows: Mapping[str, object]) -> list[tuple[int, object]]:
        """Return the signed terms of the net rate of a stock that no transfer holds."""
        return [
            (sign, term)
            for sign, term in self._split_terms(self.stocks[stock].flow, 1)
            if not (self._is_flow_variable(term) and self.get_reference(term) in flows)
        ]

    def _reads_stock(self, ast: object, stock: str) -> bool:
        """Check that an expression reads a stock, directly or through auxiliaries."""
        return self._holds(ast, lambda item: self.get_reference(item) == stock, set())

    def get_goal_seeking(self, stock: str, terms: Sequence[tuple[int, object]]) -> _GoalSeekingSpec | None:
        """Return the goal-seeking form of the remaining terms of a stock.

        It returns `None` for any other form.
        """
        if len(terms) != 1 or terms[0][0] < 0 or not isinstance(terms[0][1], ArithmeticStructure):
            return None
        term = terms[0][1]
        if term.operators not in (['/'], ['*']) or not isinstance(term.arguments[0], ArithmeticStructure):
            return None
        gap, scale = term.arguments
        if gap.operators[-1:] != ['-'] or self.get_reference(gap.arguments[-1]) != stock:
            return None
        rest = gap.operators[:-1]
        goal = ArithmeticStructure(rest, gap.arguments[:-1]) if rest else gap.arguments[0]
        if self._reads_stock(goal, stock) or self._reads_stock(scale, stock):
            return None
        return _GoalSeekingSpec(goal, scale, term.operators == ['/'])

    def _add_ordered(self, name: str, ordered: list[str], visiting: list[str], problems: list[str]) -> None:
        """Add an auxiliary after those it reads, and collect one that reads itself."""
        if name in ordered:
            return
        if name in visiting:
            problems.append(f'{self.originals[name]}: an auxiliary that reads itself')
            return
        visiting.append(name)
        stack = [component.ast for component in self.auxiliaries[name]]
        while stack:
            ast = stack.pop()
            stack.extend(self._get_children(ast))
            if self.get_reference(ast) in self.auxiliaries:
                self._add_ordered(self.get_reference(ast), ordered, visiting, problems)
        visiting.remove(name)
        ordered.append(name)

    def get_order_auxiliaries(self, problems: list[str]) -> list[str]:
        """Return the auxiliaries so that each comes after those it reads.

        It collects those that read themselves.
        """
        ordered: list[str] = []
        for name in sorted(self.auxiliaries):
            self._add_ordered(name, ordered, [], problems)
        return ordered

    def _get_read_by_laws(self) -> set[str]:
        """Return the auxiliaries that the net rate of a stock reads.

        It reads them directly or through other auxiliaries.
        """
        read: set[str] = set()
        for integral in self.stocks.values():
            self._holds(integral.flow, lambda _: False, read)
        return read

    def is_observable(self, name: str) -> bool:
        """Check that a run can record an auxiliary.

        It holds no fixed delay and no delay that no law reads.
        """
        ast = self.auxiliaries[name][0].ast
        if self._holds(ast, lambda item: isinstance(item, DelayFixedStructure), set()):
            return False
        return name in self._get_read_by_laws() or not self._holds(ast, lambda item: isinstance(item, DELAYS), set())

    def is_per_member(self, name: str) -> bool:
        """Check that an auxiliary has one value per member of the group."""
        return any(component.subscripts[0] for component in self.auxiliaries.get(name, ()))

    def get_covered(self, component: AbstractComponent) -> list[int] | None:
        """Return the members a component of a variable defines.

        It returns `None` for subscripts it cannot be a group of.
        """
        subscripts, excepted = component.subscripts
        if excepted or len(subscripts) > 1:
            return None
        if not subscripts or subscripts[0] == self.range_name:
            return list(range(len(self.members) or 1))
        return [self.members.index(subscripts[0])] if subscripts[0] in self.members else None
