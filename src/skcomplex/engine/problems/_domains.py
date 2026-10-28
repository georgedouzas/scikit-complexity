"""Read the Sage sets that states and parameters take their values in: their kind, and their stored numbers."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import CC, NN, ZZ
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ._kinds import DomainKind

_BOOLEAN_LABELS = (frozenset({0, 1}), frozenset({False, True}))


def wrap_angle(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return angles in `(-pi, pi]`, with `-pi` moved to `pi` and angles already inside left exactly as they are.

    Args:
        values:
            An angle or an array of angles in radians.

    Returns:
        angles:
            The wrapped angles.

    Examples:
        >>> import numpy as np
        >>> from skcomplex.engine.problems import wrap_angle
        >>> np.round(wrap_angle([-np.pi / 2, -np.pi, 0.5]), 4).tolist()
        [-1.5708, 3.1416, 0.5]
    """
    angles = np.asarray(values, dtype=float)
    inside = (angles > -np.pi) & (angles <= np.pi)
    return np.where(inside, angles, np.pi - np.mod(np.pi - angles, 2 * np.pi))


def _is_number(item: object) -> bool:
    """Check whether an element of a finite set is a number, which a state stores as it is."""
    return isinstance(item, int | float) or getattr(item, 'is_numeric', lambda: False)()


def labels_of(domain: object) -> tuple[object, ...] | None:
    """Return the labels of a finite domain stored as indices, or `None` for a numeric or a boolean domain.

    A finite set of numbers, such as the spins `FiniteEnumeratedSet([-1, 1])`, stores its values as they are.

    Args:
        domain:
            The Sage set of the values.

    Returns:
        labels:
            The labels in order, or `None`.

    Examples:
        >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
        >>> from skcomplex.engine.problems import labels_of
        >>> labels_of(FiniteEnumeratedSet(['S', 'I', 'R']))
        ('S', 'I', 'R')
    """
    if not isinstance(domain, FiniteEnumeratedSet) or frozenset(domain) in _BOOLEAN_LABELS:
        return None
    if all(_is_number(item) for item in domain):
        return None
    return tuple(domain)


def domain_kind(domain: object, period: object = None) -> DomainKind:
    """Return the kind of value a Sage set holds, with an angle for a periodic state.

    Args:
        domain:
            The Sage set of the values.

        period:
            The period the values wrap around, or `None`.

    Returns:
        kind:
            The kind of value.

    Examples:
        >>> from sage.all__sagemath_symbolics import RR, ZZ
        >>> from skcomplex.engine.problems import domain_kind
        >>> domain_kind(ZZ), domain_kind(RR, period=6.28)
        (<DomainKind.INTEGER: 'integer'>, <DomainKind.ANGLE: 'angle'>)
    """
    if period is not None:
        return DomainKind.ANGLE
    if domain is CC:
        return DomainKind.COMPLEX
    if isinstance(domain, FiniteEnumeratedSet):
        if labels_of(domain) is not None:
            return DomainKind.CATEGORICAL
        return DomainKind.BOOLEAN if frozenset(domain) in _BOOLEAN_LABELS else DomainKind.INTEGER
    if domain is ZZ or domain is NN:
        return DomainKind.INTEGER
    return DomainKind.REAL


def _sage_assumption(domain: object) -> str:
    """Return the Sage domain of the symbols of a Sage set: real, positive, integer, or complex.

    Args:
        domain:
            The Sage set of the values.

    Returns:
        assumption:
            The Sage domain of the symbols.
    """
    kind = domain_kind(domain)
    if kind == DomainKind.COMPLEX:
        return 'complex'
    if kind in {DomainKind.INTEGER, DomainKind.BOOLEAN, DomainKind.CATEGORICAL}:
        return 'integer'
    if isinstance(domain, RealSet) and domain.inf() >= 0 and 0 not in domain:
        return 'positive'
    return 'real'


def encode(domain: object, value: object, period: object = None) -> npt.NDArray[np.generic]:
    """Return a value as the stored numbers of a state vector.

    Labels become their indices, booleans 0 and 1, angles their values in `(-pi, pi]`, and a complex domain keeps
    complex numbers. A real domain takes the real part of a complex number.

    Args:
        domain:
            The Sage set of the values.

        value:
            A value or an array of values.

        period:
            The period the values wrap around, or `None`.

    Returns:
        array:
            The stored numbers.

    Examples:
        >>> from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
        >>> from skcomplex.engine.problems import encode
        >>> encode(FiniteEnumeratedSet(['low', 'high']), ['high', 'low']).tolist()
        [1.0, 0.0]
    """
    labels = labels_of(domain)
    if labels is not None:
        array = np.asarray(value, dtype=object)
        return np.vectorize(lambda item: float(labels.index(item) if item in labels else item), otypes=[float])(array)
    array = np.asarray(value)
    if domain is CC:
        return np.asarray(array, dtype=complex)
    array = np.asarray(array.real if np.iscomplexobj(array) else array, dtype=float)
    return wrap_angle(array) if period is not None else array


def decode(domain: object, values: npt.ArrayLike) -> npt.NDArray[np.generic]:
    """Return stored numbers as values of a Sage set: labels for a categorical domain, booleans for a boolean one.

    Args:
        domain:
            The Sage set of the values.

        values:
            The stored numbers.

    Returns:
        values:
            The values.
    """
    labels = labels_of(domain)
    if labels is not None:
        indices = np.rint(np.asarray(values)).astype(int)
        return np.asarray(labels, dtype=object)[np.clip(indices, 0, len(labels) - 1)]
    if domain_kind(domain) == DomainKind.BOOLEAN:
        return np.asarray(values) > 0.5  # noqa: PLR2004
    return np.asarray(values)


def contains(domain: object, value: object) -> bool:
    """Check that every component of a value belongs to a Sage set.

    Args:
        domain:
            The Sage set of the values.

        value:
            A value or an array of values.

    Returns:
        is_contained:
            Whether every component belongs to the set.
    """
    labels = labels_of(domain)
    items = np.asarray(value, dtype=object).ravel()
    if labels is not None:
        return all(item in labels or (isinstance(item, int) and 0 <= item < len(labels)) for item in items)
    try:
        return all(complex(item) in domain if domain is CC else item in domain for item in items)
    except (TypeError, ValueError):
        return False
