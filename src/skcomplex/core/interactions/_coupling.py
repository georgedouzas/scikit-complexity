"""Define coupling, which gives every member a reduction over other members of a function of both of them."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated, cast

import numpy as np
import numpy.typing as npt
import scipy.sparse
from sage.all__sagemath_symbolics import SR, Expression, arctan2, cos, oo, sin
from sage.modules.free_module_element import FreeModuleElement
from sage.sets.real_set import RealSet

from ...errors import ValidationError
from ...specs import ParameterSpec, RoleSpec
from ..laws import BaseLaw, RateLaw, UpdateLaw
from ..laws.utils import count, delayed, maximum, minimum, neighbors, shift_member, total
from ..spaces import BaseNeighborhood
from ..types import Combination, NamespaceFunction, Target
from ._neighbor import NeighborInteraction

MODES = ('edges', 'group', 'next')
REDUCTIONS = ('sum', 'mean', 'max', 'min', 'normalized-mean', 'circular-mean')
SUMS = ('sum', 'mean')
EXTREMES = ('max', 'min')
FORMS = ('rate', 'map')
ENDS = ('source', 'target')

type Weights = str | npt.ArrayLike | scipy.sparse.sparray | NamespaceFunction | None
type Part = Callable[[Target, Target], Target]


class CouplingInteraction(NeighborInteraction):
    """Coupling of every member to other members, through a reduction over them of a function of both members.

    Every member `i` gets the reduction over other members `j` of `function(namespace)`, where `namespace.member` reads
    the states of `i`, `namespace.other` those of `j`, `namespace.edge` the member of the edge group that joins them, or
    `None`, and the namespace also holds the parameters and the time `t`. The function is the difference `x_j - x_i` of
    the state `read` by default. The members `j` are, by `mode`:

    - `'edges'`: the neighbours of `i` along the edges of its lattice or network, or within the `neighborhood`.
    - `'group'`: every other member of the group, or every member of the group `source` when it is given.

    In the mode `'group'` the roles may also bind separate elements. A target that lists elements couples every element
    to the other elements of the list, a single target element is a hub that reduces over every member of the group
    `source`, and a source that is one element or a list of elements couples every member of the target to them. The
    reduction over separate elements is written out term by term.
    - `'next'`: the next member of the group in member order, the first after the last.

    With `include_self`, the reduction also runs over `j = i`. Every term `f_ij` has a weight `w_ij`: on edges, the
    state `weights` of the edge group, which a subclass binds with a second role, over the group, the entry `W_ij` of
    the matrix `weights`, and in every mode a callable `weights(namespace)` of the namespace of the term, such as an
    indicator of the states of both members. The reductions are `'sum'`, `sum_j w_ij f_ij`, `'mean'`, the sum divided
    by the number of members `j`, `'max'` and `'min'`, the largest and smallest `w_ij f_ij`, `'normalized-mean'`,
    `sum_j w_ij f_ij / sum_j w_ij`, and `'circular-mean'`, the angle `arctan2(sum_j w_ij sin f_ij, sum_j w_ij cos
    f_ij)`. The state `read` of `j` is read `delay` earlier, and the one of `i` shifted by the phase lag `lag`, when
    they are not 0.

    In the form `'rate'` the reduction adds to the rate of the state `write`, and in the form `'map'` it sets the next
    value of `write` every step of discrete time to `combine(namespace, reduction)`, where `namespace.member` reads the
    states of `i` and the namespace holds the parameters and the time, or to the reduction itself by default. The sum
    and the mean of the rate on the edges of a lattice or network run once per edge along both directions, and every
    other coupling runs once per member.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of the members, or in the mode `'group'` the path of one element or the paths of
            separate elements.

        source:
            The path of the group, the element, or the paths of the separate elements of the other members in the mode
            `'group'`, or `None` for the members of the target.

        read:
            The name of the state that the default function, the delay, and the lag read.

        write:
            The name of the state that the coupling changes.

        function:
            The term of one other member, a callable of the namespace, or `None` for the difference of `read`.

        reduce:
            `'sum'`, `'mean'`, `'max'`, `'min'`, `'normalized-mean'`, or `'circular-mean'`.

        mode:
            `'edges'`, `'group'`, or `'next'`.

        weights:
            The name of the state of the edge group in the mode `'edges'`, a matrix with one row per member and one
            column per other member in the mode `'group'`, a callable of the namespace of a term in any mode, or `None`
            for no weights.

        include_self:
            Whether the reduction also runs over the member itself.

        lag:
            The phase lag added to the state `read` of the member.

        delay:
            The delay of the state `read` of the other member, 0 for none.

        form:
            `'rate'` or `'map'`.

        combine:
            The next value in the form `'map'`, a callable of the namespace of the member and the reduction, or `None`
            for the reduction.

        neighborhood:
            The neighbourhood of the mode `'edges'`, such as `RadiusNeighborhood(1.0)` in a continuous space, or `None`
            for the edges of the lattice or network.

        layer:
            The layer of the network whose edges the interaction runs on, or `None` for the graph of the network.

    Raises:
        ValidationError:
            A variant is unknown, the function or the combination is not callable, or the weights, the source, the
            reduction, or the neighbourhood do not suit the mode and the form, raised by `check`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import sin, var
        >>> from skcomplex.core.interactions import CouplingInteraction
        >>> source, target = SimpleNamespace(theta=var('a')), SimpleNamespace(theta=var('b'))
        >>> sine = lambda s: sin(s.other.theta - s.member.theta)
        >>> coupling = CouplingInteraction('coupling', target='oscillators', read='theta', write='theta', function=sine)
        >>> [law.value for law in coupling.check().prepare().run(SimpleNamespace(source=source, target=target)).laws_]
        [sin(a - b), sin(-a + b)]
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str | Sequence[str], RoleSpec()],
        source: Annotated[str | Sequence[str] | None, RoleSpec(optional=True)] = None,
        read: str = 'x',
        write: str = 'x',
        function: NamespaceFunction | None = None,
        reduce: str = 'sum',
        mode: str = 'edges',
        weights: Weights = None,
        include_self: bool = False,
        lag: Annotated[float, ParameterSpec()] = 0.0,
        delay: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        form: str = 'rate',
        combine: Combination | None = None,
        neighborhood: BaseNeighborhood | None = None,
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.target = target
        self.source = source
        self.read = read
        self.write = write
        self.function = function
        self.reduce = reduce
        self.mode = mode
        self.weights = weights
        self.include_self = include_self
        self.lag = lag
        self.delay = delay
        self.form = form
        self.combine = combine
        self.neighborhood = neighborhood

    def _check_variants(self) -> None:
        """Raise an error for an unknown mode, reduction, or form, or a function or combination that is not callable."""
        for name, value, known in (
            ('mode', self.mode, MODES),
            ('reduce', self.reduce, REDUCTIONS),
            ('form', self.form, FORMS),
        ):
            if value not in known:
                error_msg = f'The {name} of coupling is {value!r}. Use one of {", ".join(known)}.'
                raise ValidationError(error_msg)
        for role, item in (('function', self.function), ('combine', self.combine)):
            if item is not None and not callable(item):
                error_msg = f'The {role} of coupling must be callable or None, but it is {item!r}.'
                raise ValidationError(error_msg)

    def _check_reduction(self) -> None:
        """Raise an error for a source outside the mode group, or a reduction that its mode cannot run."""
        if self.source is not None and self.mode != 'group':
            error_msg = f'The source {self.source!r} of coupling needs the mode group, but the mode is {self.mode!r}.'
            raise ValidationError(error_msg)
        if not isinstance(self.target, str) and self.mode != 'group':
            error_msg = (
                f'The coupling of the separate elements {self.target!r} needs the mode group, not {self.mode!r}.'
            )
            raise ValidationError(error_msg)
        if self.mode == 'next' and self.reduce not in SUMS:
            error_msg = f'The coupling to the next member reduces by sum or mean, not {self.reduce!r}.'
            raise ValidationError(error_msg)
        if self.reduce in EXTREMES and self.mode == 'group' and self._excludes_self() and isinstance(self.target, str):
            error_msg = (
                f'The {self.reduce} of coupling over its own group runs over every member. Pass include_self=True or '
                'a separate source group.'
            )
            raise ValidationError(error_msg)
        if self.reduce in EXTREMES and self._has_matrix():
            error_msg = f'The {self.reduce} of coupling takes no matrix of weights. Give the weights as a callable.'
            raise ValidationError(error_msg)

    def _check_structure(self) -> None:
        """Raise an error for weights or a neighbourhood that do not suit the mode and the form."""
        if self.neighborhood is not None and (
            self.mode != 'edges' or not isinstance(self.neighborhood, BaseNeighborhood)
        ):
            error_msg = f'The neighbourhood {self.neighborhood!r} of coupling needs a neighbourhood in the mode edges.'
            raise ValidationError(error_msg)
        if isinstance(self.weights, str) and not self._is_per_edge():
            error_msg = (
                f'The weights {self.weights!r} of coupling name a state of the edges, which only the sum or mean rate '
                'on the edges of a lattice or network reads. Give a matrix in the mode group, or a callable.'
            )
            raise ValidationError(error_msg)
        if self._has_matrix() and self.mode != 'group':
            error_msg = 'A matrix of weights of coupling needs the mode group. Name an edge state on the edges.'
            raise ValidationError(error_msg)

    def _check(self) -> None:
        """Raise an error for unknown variants, functions that cannot be called, or weights that do not suit the mode.

        Raises:
            ValidationError:
                A variant is unknown, the function or the combination is not callable, or the weights, the source, the
                reduction, or the neighbourhood do not suit the mode and the form.
        """
        super()._check()
        self._check_variants()
        self._check_reduction()
        self._check_structure()

    def _has_matrix(self) -> bool:
        """Return whether the weights are a matrix."""
        return self.weights is not None and not isinstance(self.weights, str) and not callable(self.weights)

    def _excludes_self(self) -> bool:
        """Return whether a reduction over the own group leaves the member out."""
        return self.source is None and not self.include_self

    def _is_per_edge(self) -> bool:
        """Return whether the coupling is a rate written once per edge of a lattice or network."""
        return self.mode == 'edges' and self.form == 'rate' and self.neighborhood is None and self.reduce in SUMS

    @staticmethod
    def _parameters(namespace: SimpleNamespace) -> dict[str, object]:
        """Return the parameters and the time of a namespace, without the ends and the edge."""
        return {name: value for name, value in vars(namespace).items() if name not in {*ENDS, 'edge'}}

    def _get_term(
        self,
        namespace: SimpleNamespace,
        member: SimpleNamespace,
        other: SimpleNamespace,
        edge: SimpleNamespace | None = None,
    ) -> Target:
        """Return the term of one other member, with the delay and the lag applied to the read state."""
        member_values, other_values = dict(vars(member)), dict(vars(other))
        if self.delay != 0:
            other_values[self.read] = delayed(other_values[self.read], namespace.delay)
        if self.lag != 0:
            member_values[self.read] = member_values[self.read] + namespace.lag
        if self.function is None:
            return other_values[self.read] - member_values[self.read]
        return self.function(
            SimpleNamespace(
                **self._parameters(namespace),
                member=SimpleNamespace(**member_values),
                other=SimpleNamespace(**other_values),
                edge=edge,
            ),
        )

    def _get_weight(
        self,
        namespace: SimpleNamespace,
        member: SimpleNamespace,
        other: SimpleNamespace,
        edge: SimpleNamespace | None = None,
    ) -> Target:
        """Return the weight of the term of one other member from callable weights, or 1 for other weights."""
        if not callable(self.weights):
            return SR(1)
        return self.weights(SimpleNamespace(**self._parameters(namespace), member=member, other=other, edge=edge))

    def _get_matrix(self) -> npt.ArrayLike | scipy.sparse.sparray | None:
        """Return the matrix of weights, with a zero diagonal unless the member is included, or `None`."""
        if not self._has_matrix():
            return None
        weights = cast(npt.ArrayLike | scipy.sparse.sparray, self.weights)
        if not self._excludes_self():
            return weights
        if scipy.sparse.issparse(weights):
            matrix = scipy.sparse.csr_array(weights, dtype=float)
            matrix.setdiag(0)
            matrix.eliminate_zeros()
            return matrix
        matrix = np.array(weights, dtype=float)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    @staticmethod
    def _map(value: Target, operation: Callable[[Expression], Expression]) -> Target:
        """Return an operation applied to an expression, or to every component of a vector."""
        if isinstance(value, FreeModuleElement):
            return value.parent()([operation(SR(item)) for item in value])
        return operation(SR(value))

    @staticmethod
    def _shifted(member: SimpleNamespace) -> SimpleNamespace:
        """Return the namespace of the next member of the group in member order."""
        return SimpleNamespace(
            **{
                name: CouplingInteraction._map(value, lambda item: shift_member(item, 1))
                for name, value in vars(member).items()
                if isinstance(value, Expression | FreeModuleElement)
            },
        )

    def _get_reduced(self, total_of: Callable[[Part], Target], number: Target) -> Target:
        """Return the reduction from sums of parts of the weighted terms and the number of members, by reduction."""
        if self.reduce == 'sum':
            return total_of(lambda value, weight: weight * value)
        if self.reduce == 'mean':
            return total_of(lambda value, weight: weight * value) / number
        if self.reduce == 'normalized-mean':
            return total_of(lambda value, weight: weight * value) / total_of(lambda value, weight: weight)
        sines = total_of(lambda value, weight: weight * sin(value))
        return arctan2(sines, total_of(lambda value, weight: weight * cos(value)))

    def _get_group_total(self, namespace: SimpleNamespace, part: Part) -> Target:
        """Return the sum of a part of the weighted term over the other members of the group."""
        source, target = namespace.source, namespace.target
        matrix = self._get_matrix()
        value, weight = self._get_term(namespace, target, source), self._get_weight(namespace, target, source)
        reduction = self._map(part(value, weight), lambda item: total(item, over=source, weights=matrix))
        if self._excludes_self() and matrix is None:
            reduction = reduction - part(
                self._get_term(namespace, target, target),
                self._get_weight(namespace, target, target),
            )
        return reduction

    @staticmethod
    def _get_extreme(values: list[Target], extreme: Callable[..., Expression]) -> Target:
        """Return the largest or smallest of several values, component by component for vectors."""
        if len(values) == 1:
            return values[0]
        first = values[0]
        if isinstance(first, FreeModuleElement):
            columns = zip(*[list(value) for value in values], strict=True)
            return first.parent()([extreme(*[SR(item) for item in column]) for column in columns])
        return extreme(*[SR(value) for value in values])

    @staticmethod
    def _get_sum(pairs: list[tuple[Target, Target]], part: Part) -> Target:
        """Return the sum of a part of the weighted terms of separate elements."""
        parts = [part(value, weight) for value, weight in pairs]
        return sum(parts[1:], parts[0])

    def _get_separate_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the separate elements of `namespace.source`, written out term by term."""
        if self._has_matrix():
            error_msg = (
                'A matrix of weights of coupling needs groups. Give the weights of separate elements as a callable.'
            )
            raise ValidationError(error_msg)
        target = namespace.target
        others = [*namespace.source, *([target] if self.include_self else [])]
        pairs = [
            (self._get_term(namespace, target, other), self._get_weight(namespace, target, other)) for other in others
        ]
        if not pairs:
            error_msg = f'The coupling {self.label!r} of separate elements has no other element. Bind two or more.'
            raise ValidationError(error_msg)
        if self.reduce in EXTREMES:
            extreme = maximum if self.reduce == 'max' else minimum
            return self._get_extreme([value * weight for value, weight in pairs], extreme)
        return self._get_reduced(lambda part: self._get_sum(pairs, part), SR(len(pairs)))

    def _get_group_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over every other member of the group, or every member of the source group."""
        source, target = namespace.source, namespace.target
        if isinstance(source, tuple):
            return self._get_separate_reduction(namespace)
        if self.reduce in EXTREMES:
            extreme = maximum if self.reduce == 'max' else minimum
            value = self._get_term(namespace, target, source) * self._get_weight(namespace, target, source)
            return self._map(value, lambda item: extreme(item, over=source))
        number = count(over=source) - (1 if self._excludes_self() else 0)
        return self._get_reduced(lambda part: self._get_group_total(namespace, part), number)

    def _get_next_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the next member of the group."""
        target = namespace.target
        shifted = self._shifted(target)
        reduction = self._get_term(namespace, target, shifted) * self._get_weight(namespace, target, shifted)
        if self.include_self:
            own = self._get_term(namespace, target, target) * self._get_weight(namespace, target, target)
            reduction = reduction + own
        return reduction / 2 if self.reduce == 'mean' and self.include_self else reduction

    def _get_neighbor_total(self, namespace: SimpleNamespace, part: Part) -> Target:
        """Return the sum of a part of the weighted term over the neighbours and the member itself."""
        source, target = namespace.source, namespace.target
        value, weight = self._get_term(namespace, target, source), self._get_weight(namespace, target, source)
        reduction = self._map(part(value, weight), lambda item: neighbors(item, over=self.neighborhood))
        return reduction + part(self._get_term(namespace, target, target), self._get_weight(namespace, target, target))

    def _get_neighbor_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction over the neighbours of the member."""
        source, target = namespace.source, namespace.target
        term = self._get_term(namespace, target, source)
        weight = self._get_weight(namespace, target, source) if callable(self.weights) else None
        if not self.include_self:
            return self._map(
                term,
                lambda item: neighbors(item, over=self.neighborhood, reduce=self.reduce, weight=weight),
            )
        if self.reduce in EXTREMES:
            extreme = maximum if self.reduce == 'max' else minimum
            others = neighbors(term, over=self.neighborhood, reduce=self.reduce, weight=weight)
            own = self._get_term(namespace, target, target) * self._get_weight(namespace, target, target)
            return extreme(others, own)
        number = neighbors(SR(1), over=self.neighborhood, reduce='count') + 1
        return self._get_reduced(lambda part: self._get_neighbor_total(namespace, part), number)

    def _get_reduction(self, namespace: SimpleNamespace) -> Target:
        """Return the reduction that one member gets, by mode."""
        if self.mode == 'group':
            return self._get_group_reduction(namespace)
        if self.mode == 'next':
            return self._get_next_reduction(namespace)
        return self._get_neighbor_reduction(namespace)

    def _get_edge_term(self, namespace: SimpleNamespace, member: SimpleNamespace, other: SimpleNamespace) -> Target:
        """Return the term one end of an edge gets from the other end, weighted and divided for a mean."""
        edge = getattr(namespace, 'edge', None)
        term = self._get_term(namespace, member, other, edge)
        if isinstance(self.weights, str):
            term = term * getattr(edge, self.weights)
        elif callable(self.weights):
            term = term * self._get_weight(namespace, member, other, edge)
        if self.reduce == 'mean':
            term = term / (member.degree + int(self.include_self))
        return term

    def _get_edge_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of an edge, with the rate of the member itself when it is included."""
        source, target = namespace.source, namespace.target
        laws: list[BaseLaw] = [
            RateLaw(getattr(target, self.write), self._get_edge_term(namespace, target, source)),
            RateLaw(getattr(source, self.write), self._get_edge_term(namespace, source, target)),
        ]
        if self.include_self:
            own = self._get_term(namespace, target, target) * self._get_weight(namespace, target, target)
            divisor = target.degree + 1 if self.reduce == 'mean' else 1
            laws.append(RateLaw(getattr(target, self.write), own / divisor))
        return laws

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the rates of both ends of every edge, or the rate or next value of every member.

        Args:
            namespace:
                The namespace of an edge, with the ends `source` and `target`.

        Returns:
            laws:
                The laws of the coupling.
        """
        if self._is_per_edge():
            return self._get_edge_laws(namespace)
        target = namespace.target
        reduction = self._get_reduction(namespace)
        if self.form == 'rate':
            return [RateLaw(getattr(target, self.write), reduction)]
        state = getattr(target, self.write)
        if self.combine is None:
            return [UpdateLaw(state, reduction)]
        member = SimpleNamespace(**self._parameters(namespace), member=target)
        return [UpdateLaw(state, self.combine(member, reduction))]
