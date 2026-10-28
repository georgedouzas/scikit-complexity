"""Name the Wiener processes that noise laws draw, and the covariance of their increments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Any

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ...core.errors import ValidationError
from ...core.laws import NoiseLaw
from ...core.specs import WienerProcessSpec
from ..problems import components_of
from ._declarations import Declarations
from ._specs import KernelSpec

type ProcessKey = tuple[str, str, Any]


def _process_key(spec: KernelSpec, item: object, declarations: Declarations) -> ProcessKey:
    """Return the process a label or a handle of a noise law names, as a local label or a Wiener process."""
    if isinstance(item, str):
        return ('local', spec.owner, item)
    found = declarations.handles.get(item) if isinstance(item, Expression) else None
    if found is None:
        error_msg = (
            f'A noise law of {spec.owner} names the process {item!r}, which is not a label or the handle `dW` of a '
            '`WienerProcessSpec` bound to a role.'
        )
        raise ValidationError(error_msg)
    return ('wiener', *found)


def _process_keys(spec: KernelSpec, law: NoiseLaw, declarations: Declarations) -> list[ProcessKey | None]:
    """Return the process of every component of the target of a noise law, `None` for an independent process."""
    count = len(components_of(law._get_targets()[0]))
    process = law.process
    if process is None:
        return [None] * count
    items = list(process) if isinstance(process, tuple) else components_of(process)
    if len(items) == 1:
        items = items * count
    if len(items) != count:
        error_msg = (
            f'A noise law of {spec.owner} names {len(items)} processes for a target of {count} components. Name one '
            'process, or one per component.'
        )
        raise ValidationError(error_msg)
    return [_process_key(spec, item, declarations) for item in items]


def _process_covariance(first: object, second: object, wieners: Mapping[str, WienerProcessSpec]) -> float:
    """Return the covariance per unit time of the increments of two processes, keys or independent markers."""
    if not (isinstance(first, tuple) and isinstance(second, tuple) and first[0] == 'wiener' == second[0]):
        return 1.0 if first == second else 0.0
    if first[1] != second[1]:
        return 0.0
    covariance = wieners[first[1]].covariance
    if covariance is None:
        return 1.0 if first[2] == second[2] else 0.0
    return float(np.asarray(covariance, dtype=float)[first[2], second[2]])


