"""Define pseudo-Riemannian spaces.

They are coordinate charts with a symbolic metric and the curvature it derives.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
from einsteinpy.symbolic import ChristoffelSymbols, MetricTensor, RicciScalar, RicciTensor, RiemannCurvatureTensor
from sage.all__sagemath_symbolics import SR, Expression, function, matrix
from sage.structure.element import Matrix
from sage.symbolic.function import BuiltinFunction, SymbolicFunction
from sage.symbolic.operators import FDerivativeOperator

from ...errors import ValidationError
from ..base import BasePart
from ._continuous import ContinuousSpace

type Christoffel = tuple[tuple[tuple[Expression, ...], ...], ...]
type Riemann = tuple[tuple[tuple[tuple[Expression, ...], ...], ...], ...]
type Derivation = tuple[Matrix, tuple[Expression, ...], bool]


class PseudoRiemannianSpace(ContinuousSpace):
    """Chart of coordinates with a symbolic metric `g_(mu nu)`.

    The connection and the curvature follow from the metric. The metric is a symmetric, nondegenerate Sage matrix whose
    entries are expressions of the coordinates, numbers, and symbolic functions of the coordinates. A function stands
    for an unknown of an ansatz, such as the scale factor `a(t)` of a homogeneous universe, which `EinsteinInteraction`
    solves for. The space is an unbounded continuous space with one dimension per coordinate, so a point on it has one
    component per coordinate, as a test particle of `GeodesicInteraction` does.

    `prepare` derives, through EinsteinPy, the Christoffel symbols
    `Γ^mu_(alpha beta) = g^(mu lambda) (∂_alpha g_(lambda beta) + ∂_beta g_(lambda alpha) - ∂_lambda g_(alpha beta))/2`
    and, with `curvature`, the Riemann tensor `R^rho_(sigma mu nu)`, its contraction the Ricci tensor
    `R_(mu nu) = R^rho_(mu rho nu)`, the Ricci scalar `R = g^(mu nu) R_(mu nu)`, and the Einstein tensor
    `G_(mu nu) = R_(mu nu) - R g_(mu nu) / 2`. The curvature of a metric with many entries, such as Kerr's, takes long
    to simplify, so it can be left out when only the connection is needed.

    Args:
        metric:
            The metric, a square matrix of expressions, or `None` for the metric a subclass derives.

        coordinates:
            The coordinates, Sage variables or their names, in the order of the rows of the metric.

        curvature:
            Whether `prepare` derives the curvature tensors besides the Christoffel symbols.

        label:
            The label of the space in its system.

    Attributes:
        coordinates_:
            The coordinates, as Sage variables.

        metric_:
            The metric `g_(mu nu)`.

        inverse_metric_:
            The inverse metric `g^(mu nu)`.

        christoffel_:
            The Christoffel symbols, with `christoffel_[mu][alpha][beta]` the symbol `Γ^mu_(alpha beta)`.

        riemann_:
            The Riemann tensor, with `riemann_[rho][sigma][mu][nu]` the component `R^rho_(sigma mu nu)`, or `None`
            without `curvature`.

        ricci_:
            The Ricci tensor `R_(mu nu)`, or `None` without `curvature`.

        ricci_scalar_:
            The Ricci scalar, or `None` without `curvature`.

        einstein_:
            The Einstein tensor `G_(mu nu)`, or `None` without `curvature`.

        derived_from_:
            The metric, the coordinates, and the curvature flag the tensors were derived from, which a later `prepare`
            compares by value to skip the derivation when they have not changed.

    Raises:
        ValidationError:
            The metric is not a symmetric square matrix of one row per coordinate, a coordinate is not a variable, two
            coordinates coincide, or the metric is degenerate.

    Examples:
        >>> from sage.all__sagemath_symbolics import SR, matrix, sin
        >>> from skcomplex.core.components.spaces import PseudoRiemannianSpace
        >>> theta, phi = SR.var('theta phi')
        >>> sphere = PseudoRiemannianSpace(matrix(SR, [[1, 0], [0, sin(theta) ** 2]]), (theta, phi)).prepare()
        >>> sphere.christoffel_[0][1][1]
        -cos(theta)*sin(theta)
        >>> sphere.ricci_scalar_
        2
    """

    space_tags: ClassVar[tuple[str, ...]] = ('curved spacetime',)

    def __init__(
        self,
        metric: Matrix | Sequence[Sequence[object]] | None = None,
        coordinates: Sequence[Expression | str] = (),
        *,
        curvature: bool = True,
        label: str = 'space',
    ) -> None:
        super().__init__(label=label)
        self.metric = metric
        self.coordinates = coordinates
        self.curvature = curvature

    def _get_coordinates(self) -> tuple[Expression, ...]:
        """Return the coordinates as Sage variables.

        This is the hook a subclass overrides to name its own.
        """
        return tuple(SR.var(item) if isinstance(item, str) else SR(item) for item in self.coordinates)

    def _get_metric(self) -> Matrix:
        """Return the metric as a Sage matrix.

        This is the hook a subclass overrides to derive it from its parameters.
        """
        if self.metric is None:
            error_msg = f'{type(self).__name__} has no metric. Give a square matrix of expressions.'
            raise ValidationError(error_msg)
        return matrix(SR, self.metric)

    def _walk(self, expression: Expression) -> Iterator[Expression]:
        """Yield an expression and every subexpression of it."""
        yield expression
        for operand in expression.operands():
            yield from self._walk(operand)

    def _get_functions(self, metric: Matrix) -> list[SymbolicFunction]:
        """Return the symbolic functions of the metric in order of appearance.

        They are those of the ansatz.
        """
        functions: list[SymbolicFunction] = []
        for entry in metric.list():
            for item in self._walk(SR(entry)):
                kind = item.operator()
                kind = kind.function() if isinstance(kind, FDerivativeOperator) else kind
                if isinstance(kind, SymbolicFunction) and not isinstance(kind, BuiltinFunction):
                    functions += [] if kind in functions else [kind]
        return functions

    def _rename(self, expression: Expression, pairs: Sequence[tuple[SymbolicFunction, SymbolicFunction]]) -> Expression:
        """Return an expression with every function of a pair replaced by the other."""
        for old, new in pairs:
            expression = expression.substitute_function(old, new)
        return expression

    def _to_sage(self, value: object, pairs: Sequence[tuple[SymbolicFunction, SymbolicFunction]]) -> Expression:
        """Return a SymPy expression of EinsteinPy as a Sage expression.

        The functions of the ansatz are restored.
        """
        return self._rename(SR(value), [(new, old) for old, new in pairs])

    def _is_derived(self, source: Derivation) -> bool:
        """Return whether the tensors were derived from an equal source.

        The source is the metric, the coordinates, and the curvature flag.
        """
        previous: Derivation | None = getattr(self, 'derived_from_', None)
        if previous is None or previous[2] != source[2] or len(previous[1]) != len(source[1]):
            return False
        if previous[0].dimensions() != source[0].dimensions():
            return False
        pairs = [*zip(previous[1], source[1], strict=True), *zip(previous[0].list(), source[0].list(), strict=True)]
        return all(SR(old).is_trivially_equal(SR(new)) for old, new in pairs)

    def _prepare(self, context: None = None) -> None:
        """Set the coordinates, the metric, its inverse, and the tensors.

        The tensors are derived again only when their source changed. A space is prepared again whenever a part that
        reads it is prepared, so the derivation, the slow step, runs only for a metric, coordinates, or a curvature flag
        that differ by value from those of the last derivation.

        It first raises an error for an invalid metric or coordinates that clash. It checks its label through the base
        of the parts, not the box of a continuous space, which it does not have.

        Args:
            context:
                Nothing, since a space is prepared alone.

        Raises:
            ValidationError:
                The metric is not a symmetric square matrix of one row per coordinate, a coordinate is not a variable,
                two coordinates coincide, or the metric is degenerate.
        """
        BasePart._prepare(self, context)
        coordinates = self._get_coordinates()
        if not coordinates or not all(item.is_symbol() for item in coordinates):
            error_msg = f'The coordinates {self.coordinates!r} are not variables. Give Sage variables or their names.'
            raise ValidationError(error_msg)
        if len(set(coordinates)) != len(coordinates):
            error_msg = f'The coordinates {self.coordinates!r} repeat a variable. Give distinct coordinates.'
            raise ValidationError(error_msg)
        metric = self._get_metric()
        if metric.nrows() != len(coordinates) or not metric.is_square():
            error_msg = (
                f'The metric has {metric.nrows()} rows and {metric.ncols()} columns for {len(coordinates)} '
                'coordinates. Give a square matrix of one row per coordinate.'
            )
            raise ValidationError(error_msg)
        if not all(SR(entry).expand().is_trivial_zero() for entry in (metric - metric.transpose()).list()):
            error_msg = 'The metric is not symmetric. Give a symmetric matrix.'
            raise ValidationError(error_msg)
        if SR(metric.det()).expand().is_trivial_zero():
            error_msg = 'The metric is degenerate, its determinant is identically zero. Give a nondegenerate metric.'
            raise ValidationError(error_msg)
        self.coordinates_ = self._get_coordinates()
        self.metric_ = self._get_metric()
        self.inverse_metric_ = self.metric_.inverse()
        n_dim = len(self.coordinates_)
        self.extent_ = ()
        self.n_dim_ = n_dim
        self.size_ = None
        self.shape_ = ()
        self.positions_ = np.zeros((0, n_dim))
        self.periodic_ = False
        self.default_visual_ = None
        self.regions_ = {}
        source: Derivation = (self.metric_, self.coordinates_, bool(self.curvature))
        if not self._is_derived(source):
            self._derive()
            self.derived_from_ = source

    def _derive(self) -> None:
        """Set the Christoffel symbols and, with `curvature`, the curvature tensors.

        They are derived through EinsteinPy.
        """
        n_dim = len(self.coordinates_)
        # SymPy reads some names of functions, such as `E` or `gamma`, as its own, so the ansatz is renamed on the way.
        functions = self._get_functions(self.metric_)
        pairs = [(item, function(f'skcomplex_ansatz_{index}')) for index, item in enumerate(functions)]
        rows = [[self._rename(SR(entry), pairs)._sympy_() for entry in row] for row in self.metric_.rows()]
        tensor = MetricTensor(rows, [item._sympy_() for item in self.coordinates_])
        christoffel = ChristoffelSymbols.from_metric(tensor)
        values = christoffel.tensor()
        indices = range(n_dim)
        self.christoffel_: Christoffel = tuple(
            tuple(tuple(self._to_sage(values[mu, alpha, beta], pairs) for beta in indices) for alpha in indices)
            for mu in indices
        )
        self.riemann_: Riemann | None = None
        self.ricci_: Matrix | None = None
        self.ricci_scalar_: Expression | None = None
        self.einstein_: Matrix | None = None
        if not self.curvature:
            return
        riemann = RiemannCurvatureTensor.from_christoffels(christoffel)
        values = riemann.tensor()
        self.riemann_ = tuple(
            tuple(
                tuple(tuple(self._to_sage(values[rho, sigma, mu, nu], pairs) for nu in indices) for mu in indices)
                for sigma in indices
            )
            for rho in indices
        )
        ricci = RicciTensor.from_riemann(riemann)
        values = ricci.tensor()
        self.ricci_ = matrix(SR, n_dim, n_dim, lambda mu, nu: self._to_sage(values[mu, nu], pairs))
        self.ricci_scalar_ = self._to_sage(RicciScalar.from_riccitensor(ricci).expr, pairs).simplify_full()
        self.einstein_ = (self.ricci_ - self.ricci_scalar_ * self.metric_ / 2).apply_map(
            lambda entry: SR(entry).simplify_full(),
        )
