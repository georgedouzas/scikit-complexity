"""Hold the Numba stepper loops of every form, which call the step count, and the arrays they take."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any

import numpy as np

from ..compile._backends import Lowering

NO_BOUNDARY = """
HAS_BOUNDARY = False


@njit(cache=CACHE)
def boundary(t, x, p, L):
    return x
"""
LOOPS = {
    'rk4': """
@njit(cache=CACHE)
def rk4(t, x, p, L, until, h):
    if until <= t:
        return x.copy()
    n_steps = step_count(t, until, h)
    step = (until - t) / n_steps
    y = x.copy()
    for index in range(n_steps):
        time = t + index * step
        k1 = drift(time, y, p, L)
        k2 = drift(time + step / 2, y + step / 2 * k1, p, L)
        k3 = drift(time + step / 2, y + step / 2 * k2, p, L)
        k4 = drift(time + step, y + step * k3, p, L)
        y = y + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if HAS_BOUNDARY:
            y = boundary(time + step, y, p, L)
    return y
""",
    'velocity_verlet': """
@njit(cache=CACHE)
def velocity_verlet(t, x, p, L, until, h, positions, velocities):
    if until <= t:
        return x.copy()
    n_steps = step_count(t, until, h)
    step = (until - t) / n_steps
    y = x.copy()
    rates = drift(t, y, p, L)
    for index in range(n_steps):
        time = t + (index + 1) * step
        for component in range(velocities.size):
            y[velocities[component]] += step / 2 * rates[velocities[component]]
        for component in range(positions.size):
            y[positions[component]] += step * y[velocities[component]]
        if HAS_BOUNDARY:
            y = boundary(time, y, p, L)
        rates = drift(time, y, p, L)
        for component in range(velocities.size):
            y[velocities[component]] += step / 2 * rates[velocities[component]]
    return y
""",
    'euler_maruyama': """
@njit(cache=CACHE)
def euler_maruyama(t, x, p, L, until, h, rng, rows, columns, processes, starts, sizes, offsets, factors):
    if until <= t:
        return x.copy()
    n_steps = step_count(t, until, h)
    step = (until - t) / n_steps
    scale = math.sqrt(step)
    y = x.copy()
    for index in range(n_steps):
        time = t + index * step
        amplitudes = diffusion(time, y, p, L)
        increments = rng.normal(0.0, scale, processes)
        for block in range(starts.size):
            start, size, offset = starts[block], sizes[block], offsets[block]
            independent = increments[start : start + size].copy()
            for row in range(size):
                total = 0.0
                for column in range(size):
                    total += factors[offset + row * size + column] * independent[column]
                increments[start + row] = total
        y = y + drift(time, y, p, L) * step
        for output in range(amplitudes.size):
            y[rows[output]] += amplitudes[output] * increments[columns[output]]
        if HAS_BOUNDARY:
            y = boundary(time + step, y, p, L)
    return y
""",
    'gillespie': """
@njit(cache=CACHE)
def gillespie(t, x, p, L, until, rng, targets, increments):
    y = x.copy()
    if until <= t:
        return y
    while True:
        rates = propensities(t, y, p, L)
        total = 0.0
        for channel in range(rates.size):
            if not (rates[channel] >= 0.0 and rates[channel] < np.inf):
                raise ValueError('Propensities must be finite and non-negative.')
            total += rates[channel]
        if total == 0.0:
            return y
        t += rng.exponential(1.0 / total)
        if t > until:
            return y
        channel = min(np.searchsorted(np.cumsum(rates), rng.uniform(0.0, total), side='right'), rates.size - 1)
        for change in range(targets.shape[1]):
            if targets[channel, change] >= 0:
                y[targets[channel, change]] += increments[channel, change]
        if HAS_BOUNDARY:
            y = boundary(t, y, p, L)
""",
    'iterate': """
@njit(cache=CACHE)
def iterate(t, x, p, L, until):
    y = x.copy()
    for step in range(int(round(t)), int(math.floor(until + 1e-9))):
        y = update(float(step), y, p, L)
        if HAS_BOUNDARY:
            y = boundary(float(step + 1), y, p, L)
    return y
""",
    'iterate_random': """
@njit(cache=CACHE)
def iterate_random(t, x, p, L, until, rng, offset, normal):
    y = x.copy()
    q = np.zeros(offset + normal.size)
    q[:offset] = p[:offset]
    for step in range(int(round(t)), int(math.floor(until + 1e-9))):
        for index in range(normal.size):
            if normal[index]:
                q[offset + index] = rng.standard_normal()
            else:
                q[offset + index] = rng.random()
        y = update(float(step), y, q, L)
        if HAS_BOUNDARY:
            y = boundary(float(step + 1), y, q, L)
    return y
""",
}


def _noise_arrays(lowering: Lowering) -> tuple[Any, ...]:
    """Return the rows, columns, process count, and flattened covariance blocks of the noise of a lowering."""
    blocks = lowering.noise_blocks
    sizes = np.asarray([block.size for block in blocks], dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes**2)])[:-1].astype(np.int64)
    factors = np.concatenate([np.zeros(0), *(np.ravel(block.factor) for block in blocks)]).astype(np.float64)
    return (
        lowering.noise_rows.astype(np.int64),
        lowering.noise_columns.astype(np.int64),
        int(lowering.n_processes),
        np.asarray([block.start for block in blocks], dtype=np.int64),
        sizes,
        offsets,
        factors,
    )


def _channel_arrays(lowering: Lowering) -> tuple[np.ndarray, np.ndarray]:
    """Return the state targets and increments of every jump channel, padded with -1 targets."""
    n_channels = sum(channels.size for channels in lowering.jumps)
    width = max((len(channels.increments) for channels in lowering.jumps), default=0)
    targets = np.full((n_channels, width), -1, dtype=np.int64)
    increments = np.zeros((n_channels, width))
    for channels in lowering.jumps:
        rows = slice(channels.offset, channels.offset + channels.size)
        count = len(channels.increments)
        targets[rows, :count] = channels.targets.T
        increments[rows, :count] = channels.increments
    return targets, increments
