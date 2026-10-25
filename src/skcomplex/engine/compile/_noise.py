"""Name the Wiener processes that noise laws draw, and the covariance of their increments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from ...core.base import Expression, ValidationError, components_of
from ...core.components import Wiener
from ...core.laws import NoiseLaw
from ._declarations import Declarations
from ._kernels import NoiseProcess
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
            'Wiener element bound to a role.'
        )
        raise ValidationError(error_msg)
    return ('wiener', *found)


def process_keys(spec: KernelSpec, law: NoiseLaw, declarations: Declarations) -> list[ProcessKey | None]:
    """Return the process of every component of the target of a noise law, `None` for an independent process."""
    count = len(components_of(law.target))
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


def process_covariance(first: object, second: object, wieners: Mapping[str, Wiener]) -> float:
    """Return the covariance per unit time of the increments of two processes, keys or independent markers."""
    if not (isinstance(first, tuple) and isinstance(second, tuple) and first[0] == 'wiener' == second[0]):
        return 1.0 if first == second else 0.0
    if first[1] != second[1]:
        return 0.0
    covariance = wieners[first[1]].covariance
    if covariance is None:
        return 1.0 if first[2] == second[2] else 0.0
    return float(np.asarray(covariance, dtype=float)[first[2], second[2]])


def covariance_factor(covariance: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a factor `L` with `L L^T = C` of a covariance, by Cholesky or, for a singular one, by eigenvalues."""
    matrix = np.asarray(covariance, dtype=float)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        values, vectors = np.linalg.eigh(matrix)
        return vectors * np.sqrt(np.maximum(values, 0.0))


def noise_process(key: ProcessKey | None, wieners: Mapping[str, Wiener]) -> NoiseProcess | None:
    """Return the named Wiener process of a process key, with the covariance factor of a Wiener element."""
    if key is None:
        return None
    kind, owner, item = key
    if kind == 'local':
        return NoiseProcess(key)
    wiener = wieners[owner]
    factor = None if wiener.covariance is None else covariance_factor(wiener.covariance)
    return NoiseProcess(owner, item, wiener.dimension, factor)
