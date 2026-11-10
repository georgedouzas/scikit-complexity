"""Sample plausible parameter sets of a system and match them to observed values."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from ..engine.solvers import BaseSolver
from ..errors import ExperimentError, ValidationError
from ..types import Array
from ._calibration import _check_priors
from ._runs import _check_count
from ._sweep import sweep
from .types import Bounds, Output, Source

DESIGNS = ('random', 'latin_hypercube')
MIN_PLAUSIBLE = 10


@dataclass(frozen=True)
class VirtualPopulationResult:
    """A result of `virtual_population`, the plausible rows and those selected.

    Args:
        plausible:
            The plausible rows, with one column per parameter and per output.

        acceptance:
            The share of the sampled rows that are plausible.

        selected:
            The plausible rows selected to match the target, or `None` without a target.

        inclusion:
            The probability of inclusion of every plausible row, or `None` without a target.

    Examples:
        >>> import pandas as pd
        >>> from skcomplex.core.experiments import VirtualPopulationResult
        >>> result = VirtualPopulationResult(pd.DataFrame({'r': [0.5], 'size': [2.0]}), 0.25, None, None)
        >>> len(result.plausible), result.acceptance
        (1, 0.25)
    """

    plausible: pd.DataFrame
    acceptance: float
    selected: pd.DataFrame | None
    inclusion: Array | None


def _check_constraints(constraints: Mapping[str, Sequence[float]], names: list[str]) -> dict[str, tuple[float, float]]:
    """Return the `(low, high)` range of every constrained output, after checking it.

    Args:
        constraints:
            The range of each constrained output, by the name of the output, such as `{'peak': (10.0, 50.0)}`.

        names:
            The names of the outputs.

    Returns:
        ranges:
            The `(low, high)` range of every constrained output, as floats.

    Raises:
        ValidationError:
            An output is not one of the outputs, or a range is not an increasing pair.
    """
    checked = {}
    for name, bound in constraints.items():
        if name not in names:
            error_msg = f'The constraint {name!r} names no output. The outputs are {", ".join(names)}.'
            raise ValidationError(error_msg)
        low, high = (float(item) for item in bound)
        if not low < high:
            error_msg = f'The range of the constraint {name!r} must be a pair (low, high) with low < high.'
            raise ValidationError(error_msg)
        checked[name] = (low, high)
    return checked


def _get_inclusion(observed: Array, values: Array) -> Array:
    """Return the probability of inclusion of every plausible value.

    It is the density of the observed values over the density of the plausible values, both Gaussian kernel density
    estimates, scaled so that its largest value is one.
    """
    ratio = stats.gaussian_kde(observed)(values) / stats.gaussian_kde(values)(values)
    return np.asarray(ratio / ratio.max(), dtype=float)


def virtual_population(
    system: Source,
    bounds: Bounds,
    outputs: Sequence[Output],
    constraints: Mapping[str, Sequence[float]],
    *,
    n: int = 1000,
    target: Mapping[str, npt.ArrayLike] | None = None,
    design: str = 'random',
    until: float | None = None,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
) -> VirtualPopulationResult:
    """Return plausible parameter sets of a system, and those matching observed values.

    A virtual population is a set of parameter rows whose runs look like real individuals. The function draws `n` rows
    from the priors that `bounds` give, runs the system once per row with `sweep`, and computes the outputs of every
    run. A row is plausible when every constrained output lies in its range. With `target`, the observed values of one
    output, it selects among the plausible rows with the probability of inclusion of Allen et al. (2016). This is the
    density of the observed values over the density of the plausible population at the output of the row, scaled so
    that its largest value is one, so the selected outputs follow the observed distribution.

    It appears in many fields. It samples virtual patients whose tumours and immune cells look like those of a clinical
    trial, plausible economies whose growth and inflation lie in historical ranges, and plausible ecosystems whose
    populations stay within observed counts.

    Args:
        system:
            The system, whose copies take the model parameters through `set_params`, or a factory of systems.

        bounds:
            A mapping from model parameter or dotted path to its `(low, high)` bounds or its SciPy prior.

        outputs:
            The outputs of every run, state and observable paths whose last values are the outputs, and functions of a
            trajectory that return a scalar, named by the function.

        constraints:
            A mapping from output name to its plausible `(low, high)` range.

        n:
            The number of sampled rows.

        target:
            A mapping from one output name to its observed values, or `None` for no selection.

        design:
            `'random'` or `'latin_hypercube'`.

        until:
            The end time of every run, or `None` for the tuned end time of the system.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        random_state:
            The seed of the rows, the runs, and the selection.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor that runs the rows.

    Returns:
        result:
            The plausible rows, the acceptance rate, and the selected rows and their probabilities of inclusion.

    Raises:
        ValidationError:
            The design is not a sampled one, a constraint or the target names no output, a range is not increasing,
            the target names more than one output, or `n` is not a positive integer.

        ExperimentError:
            No row is plausible, or fewer than ten rows are plausible with a target.

    Examples:
        Populations of logistic growth with a random growth rate are plausible when they reach 50 by time 5.

        >>> from skcomplex.core.experiments import virtual_population
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> result = virtual_population(
        ...     LogisticGrowthSystem(),
        ...     {'r': (0.1, 1.0)},
        ...     ['population.x'],
        ...     {'population.x': (50.0, 100.0)},
        ...     n=20,
        ...     until=5.0,
        ...     random_state=0,
        ... )
        >>> result.acceptance, bool(result.plausible['r'].min() > 0.4)
        (0.65, True)
    """
    _check_count('n', n)
    if design not in DESIGNS:
        error_msg = f'Unknown design {design!r}. A virtual population samples with one of {", ".join(DESIGNS)}.'
        raise ValidationError(error_msg)
    if target is not None and len(target) != 1:
        error_msg = f'The target holds the observed values of one output, not of {len(target)}.'
        raise ValidationError(error_msg)
    names = list(_check_priors(bounds))
    result = sweep(
        system,
        bounds,
        design=design,
        n=n,
        until=until,
        dt=dt,
        solver=solver,
        outputs=outputs,
        random_state=random_state,
        executor=executor,
    )
    frame = result.convert_to_frame()
    kept = [column for column in frame.columns if column not in {*names, 'scenario', 'replicate', 'seed'}]
    frame = frame[[*names, *kept]]
    ranges = _check_constraints(constraints, kept)
    inside = {name: frame[name].between(low, high).to_numpy() for name, (low, high) in ranges.items()}
    plausible_rows = np.logical_and.reduce([*inside.values(), np.ones(len(frame), dtype=bool)])
    if not plausible_rows.any():
        worst = min(inside, key=lambda name: int(inside[name].sum()))
        low, high = ranges[worst]
        error_msg = (
            f'No row of the virtual population is plausible. The constraint {worst!r} in ({low:g}, {high:g}) rejected '
            f'{int((~inside[worst]).sum())} of {n} rows.'
        )
        raise ExperimentError(error_msg)
    plausible = frame[plausible_rows].reset_index(drop=True)
    acceptance = len(plausible) / n
    if target is None:
        return VirtualPopulationResult(plausible, acceptance, None, None)
    [(name, observed)] = target.items()
    if name not in kept:
        error_msg = f'The target {name!r} names no output. The outputs are {", ".join(kept)}.'
        raise ValidationError(error_msg)
    if len(plausible) < MIN_PLAUSIBLE:
        error_msg = (
            f'Selection by inclusion estimates densities and needs at least {MIN_PLAUSIBLE} plausible rows, not '
            f'{len(plausible)}. Sample more rows or widen the constraints.'
        )
        raise ExperimentError(error_msg)
    inclusion = _get_inclusion(np.asarray(observed, dtype=float), plausible[name].to_numpy(dtype=float))
    rng = np.random.default_rng(np.random.SeedSequence(random_state).spawn(1)[0])
    selected = plausible[rng.random(len(plausible)) < inclusion].reset_index(drop=True)
    return VirtualPopulationResult(plausible, acceptance, selected, inclusion)
