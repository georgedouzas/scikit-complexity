"""Compile the stack walks of string rewriting: bracket-aware contexts and the turtle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt
from numba import njit

DRAW, MOVE, LEFT, RIGHT, REVERSE, PUSH, POP, PITCH_DOWN, PITCH_UP, ROLL_LEFT, ROLL_RIGHT = range(1, 12)
ACTION_CODES = {
    'draw': DRAW,
    'move': MOVE,
    'left': LEFT,
    'right': RIGHT,
    'reverse': REVERSE,
    'push': PUSH,
    'pop': POP,
    'pitch_down': PITCH_DOWN,
    'pitch_up': PITCH_UP,
    'roll_left': ROLL_LEFT,
    'roll_right': ROLL_RIGHT,
}


@njit(cache=True)
def neighbors(  # noqa: C901
    codes: npt.NDArray[np.int64],
    ignored: npt.NDArray[np.bool_],
    opening: int,
    closing: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the index of the left and the right neighbour of every module, or -1, skipping ignored symbols.

    The left neighbour of the first module of a branch is the module the branch starts from, and the right neighbour
    skips whole branches and stops at the end of a branch, as in Prusinkiewicz and Lindenmayer (1990 §1.8).
    """
    size = codes.size
    left = np.full(size, -1, dtype=np.int64)
    right = np.full(size, -1, dtype=np.int64)
    stack = np.empty(size + 1, dtype=np.int64)
    depth = 0
    last = -1
    for index in range(size):
        code = codes[index]
        if code == opening:
            stack[depth] = last
            depth += 1
        elif code == closing:
            if depth > 0:
                depth -= 1
                last = stack[depth]
        elif ignored[code]:
            left[index] = last
        else:
            left[index] = last
            last = index
    depth = 0
    following = -1
    for index in range(size - 1, -1, -1):
        code = codes[index]
        if code == closing:
            stack[depth] = following
            depth += 1
            following = -1
        elif code == opening:
            if depth > 0:
                depth -= 1
                following = stack[depth]
        elif ignored[code]:
            right[index] = following
        else:
            right[index] = following
            following = index
    return left, right


@njit(cache=True)
def _rotate(vector: npt.NDArray[np.float64], axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    """Return a vector rotated about a unit axis by an angle with the formula of Rodrigues."""
    cross = np.array(
        [
            axis[1] * vector[2] - axis[2] * vector[1],
            axis[2] * vector[0] - axis[0] * vector[2],
            axis[0] * vector[1] - axis[1] * vector[0],
        ],
    )
    dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]
    return vector * np.cos(angle) + cross * np.sin(angle) + axis * dot * (1.0 - np.cos(angle))


@njit(cache=True)
def turtle_walk(  # noqa: C901
    actions: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
    step: float,
    angle: float,
    heading: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Return the segments of shape `(k, 2, 3)` and the branch depths that a turtle draws walking a word.

    Each module has an action and a value, its first parameter or NaN, which overrides the step or the angle.
    """
    draws = 0
    for action in actions:
        if action == DRAW:
            draws += 1
    segments = np.zeros((draws, 2, 3))
    depths = np.zeros(draws, dtype=np.int64)
    position = np.zeros(3)
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    side = np.array([-np.sin(heading), np.cos(heading), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    stack = np.zeros((actions.size + 1, 4, 3))
    depth = 0
    drawn = 0
    for index in range(actions.size):
        action = actions[index]
        value = values[index]
        if action in (DRAW, MOVE):
            length = step if np.isnan(value) else value
            end = position + length * forward
            if action == DRAW:
                segments[drawn, 0] = position
                segments[drawn, 1] = end
                depths[drawn] = depth
                drawn += 1
            position = end
        elif action in (LEFT, RIGHT, REVERSE):
            turn = angle if np.isnan(value) else value
            turn = np.pi if action == REVERSE else (turn if action == LEFT else -turn)
            forward, side = _rotate(forward, up, turn), _rotate(side, up, turn)
        elif action in (PITCH_DOWN, PITCH_UP):
            turn = angle if np.isnan(value) else value
            turn = turn if action == PITCH_DOWN else -turn
            forward, up = _rotate(forward, side, turn), _rotate(up, side, turn)
        elif action in (ROLL_LEFT, ROLL_RIGHT):
            turn = angle if np.isnan(value) else value
            turn = turn if action == ROLL_LEFT else -turn
            side, up = _rotate(side, forward, turn), _rotate(up, forward, turn)
        elif action == PUSH:
            stack[depth, 0] = position
            stack[depth, 1] = forward
            stack[depth, 2] = side
            stack[depth, 3] = up
            depth += 1
        elif action == POP and depth > 0:
            depth -= 1
            position = stack[depth, 0].copy()
            forward = stack[depth, 1].copy()
            side = stack[depth, 2].copy()
            up = stack[depth, 3].copy()
    return segments, depths
