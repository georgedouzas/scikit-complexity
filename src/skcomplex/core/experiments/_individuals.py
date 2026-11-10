"""Fit the model parameters of a system to the data of every individual."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import functools
import operator
from collections.abc import Hashable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..components.systems import BaseSystem
from ..engine.solvers import BaseSolver
from ..errors import ValidationError
from ._calibration import CalibrationResult, calibrate
from ._runs import _check_source, _map_calls
from ._utils import build_system
from .types import Bounds, Source

SEED_RANGE = 2**31


type Data = Mapping[str, tuple[npt.ArrayLike, npt.ArrayLike]]


@dataclass(frozen=True)
class IndividualCalibrationResult:
    """A result of `calibrate_individuals`, the calibration of every individual.

    Args:
        calibrations:
            The calibration of every individual.

        estimates:
            The estimates, with one row per individual and one column per parameter.

    Examples:
        >>> import pandas as pd
        >>> from skcomplex.core.experiments import IndividualCalibrationResult
        >>> table = pd.DataFrame({'r': [0.5, 0.7]}, index=pd.Index(['a', 'b'], name='individual'))
        >>> IndividualCalibrationResult({}, table).estimates.loc['b', 'r']
        np.float64(0.7)
    """

    calibrations: dict[Hashable, CalibrationResult]
    estimates: pd.DataFrame


def calibrate_individuals(
    system: Source,
    data: Mapping[Hashable, Data],
    bounds: Bounds,
    *,
    known: Mapping[Hashable, Mapping[str, object]] | None = None,
    method: str = 'local',
    loss: str = 'squares',
    initial: Mapping[str, float] | None = None,
    n_samples: int = 1000,
    budget: int | None = None,
    chains: int = 4,
    replicates: int = 1,
    dt: float | None = None,
    solver: BaseSolver | None = None,
    random_state: int | None = None,
    executor: str | Executor = 'serial',
) -> IndividualCalibrationResult:
    """Return the fit of the model parameters of a system to each individual's data.

    Each individual has its own data and may have known model parameters, such as the dose a patient received. The
    function sets the known parameters of an individual on a copy of the system, or passes them to the factory, and
    calls `calibrate` on that individual's data with the parameters of `bounds`. The fitted system of an individual is
    its digital twin. The calibrations are independent, so they run on `executor`, and every individual gets its own
    seed spawned from `random_state`.

    It appears in many fields. It fits the growth and the response of the tumour of every patient of a trial, the
    rates of an epidemic in every region, and the demand of every firm of a market.

    Args:
        system:
            The system, whose copies take the model parameters through `set_params`, or a factory of systems.

        data:
            A mapping from individual to its data, a mapping from state or observable path to `(times, values)`.

        bounds:
            A mapping from model parameter to its `(low, high)` bounds or its SciPy prior, shared by all individuals.

        known:
            A mapping from individual to its known model parameters by name, or `None` for none.

        method:
            The method of `calibrate`.

        loss:
            The loss of `calibrate`.

        initial:
            The starting values of a local fit, or `None` for the medians of the priors.

        n_samples:
            The number of samples of a posterior method.

        budget:
            The budget of the method, or `None` for its default.

        chains:
            The number of chains of MCMC.

        replicates:
            The number of runs averaged per evaluation.

        dt:
            The time between frames, or `None` for the tuned one.

        solver:
            The solver, or `None` for the tuned one.

        random_state:
            The seed of the seeds of the individuals.

        executor:
            `'serial'`, `'thread'`, `'process'`, or a `concurrent.futures` executor that runs the calibrations.

    Returns:
        result:
            The calibration of every individual and the table of their estimates.

    Raises:
        ValidationError:
            There is no individual, a known parameter is also fitted or belongs to an individual without data, or
            `calibrate` rejects a setting.

        ExperimentError:
            The system is a name.

    Examples:
        Two populations grow at their own rates from their own known initial sizes.

        >>> import numpy as np
        >>> from skcomplex.core.experiments import calibrate_individuals
        >>> from skcomplex.systems.biology.ecology import LogisticGrowthSystem
        >>> times = np.arange(0.0, 21.0)
        >>> data = {
        ...     'north': {'population.x': (times, 100 / (1 + 9 * np.exp(-0.4 * times)))},
        ...     'south': {'population.x': (times, 100 / (1 + 4 * np.exp(-0.7 * times)))},
        ... }
        >>> known = {'north': {'n': 10.0}, 'south': {'n': 20.0}}
        >>> result = calibrate_individuals(LogisticGrowthSystem(), data, {'r': (0.1, 2.0)}, known=known)
        >>> result.estimates['r'].round(2).to_dict()
        {'north': 0.4, 'south': 0.7}
    """
    source = _check_source(system)
    if not data:
        error_msg = 'Calibrating individuals needs the data of at least one individual.'
        raise ValidationError(error_msg)
    known = known or {}
    for individual, values in known.items():
        if individual not in data:
            error_msg = f'The individual {individual!r} has known parameters but no data.'
            raise ValidationError(error_msg)
        fitted = sorted(set(values) & set(bounds))
        if fitted:
            error_msg = (
                f'The parameter {", ".join(fitted)} of the individual {individual!r} is both known and fitted. Remove '
                'it from `known` or from `bounds`.'
            )
            raise ValidationError(error_msg)
    sources: list[Source] = []
    for individual in data:
        values = dict(known.get(individual, {}))
        if isinstance(source, BaseSystem):
            sources.append(build_system(source, values))
        else:
            sources.append(functools.partial(source, **values))
    rng = np.random.default_rng(random_state)
    seeds = [None if random_state is None else int(seed) for seed in rng.integers(SEED_RANGE, size=len(data))]
    fits = [
        functools.partial(
            calibrate,
            individual_source,
            individual_data,
            bounds,
            method=method,
            loss=loss,
            initial=initial,
            n_samples=n_samples,
            budget=budget,
            chains=chains,
            replicates=replicates,
            dt=dt,
            solver=solver,
            random_state=seed,
        )
        for individual_source, individual_data, seed in zip(sources, data.values(), seeds, strict=True)
    ]
    calibrations = dict(zip(data, _map_calls(operator.call, fits, executor), strict=True))
    table = pd.DataFrame(
        [result.estimates for result in calibrations.values()],
        index=pd.Index(list(calibrations), name='individual'),
    )
    return IndividualCalibrationResult(calibrations, table)
