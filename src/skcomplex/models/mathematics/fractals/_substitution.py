"""Fractal models built by substitution: the Koch curve, the Cantor set, and the Sierpiński triangle and carpet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.base import ValidationError
from ....core.model import Affine, Patch, Production, Prototile, Rewrite, Substitution, System, Turtle, Word
from ....core.runtime import register_model

SHAPES = ('triangle', 'carpet')


@register_model(
    field='mathematics/fractals',
    tags=('fractal', 'L-system', 'rewriting system', 'turtle'),
    references=('Prusinkiewicz and Lindenmayer 1990', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023'),
    defaults={'until': 5, 'dt': 1},
)
def koch_curve(angle: float = 60.0) -> System:
    """Return the Koch curve as the L-system `F → F+F--F+F` drawn by a turtle.

    Every generation replaces each segment by four segments of a third of its length, with a bump of the turning angle
    in the middle. With the angle of 60 degrees the curve keeps its endpoints at `(0, 0)` and `(1, 0)`, its length is
    `(4/3)^n` at generation `n`, and its similarity dimension is `log 4 / log 3`, about 1.262.

    Args:
        angle:
            The turning angle of the turtle in degrees.

    Returns:
        system:
            The model, with the word `koch`, its symbol counts `koch.F`, and its size `koch.size`.

    Examples:
        >>> from skcomplex.models import run
        >>> run('koch_curve', settings={'until': 3}).trajectory_['koch.F'].tolist()
        [1.0, 4.0, 16.0, 64.0]
    """
    word = Word(
        'koch',
        axiom='F',
        alphabet='F+-',
        turtle=Turtle(angle=angle, step=1.0, contraction=1 / 3),
        bounds=(-0.03, -0.05, 1.03, 0.35),
    )
    return System(
        components=[word],
        interactions=[Rewrite('rules', target='koch', rules=[Production('F', 'F+F--F+F')])],
    )


@register_model(
    field='mathematics/fractals',
    tags=('fractal', 'substitution', 'rewriting system'),
    references=('Cantor 1883', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023'),
    defaults={'until': 8, 'dt': 1},
)
def cantor_set(ratio: float = 1 / 3) -> System:
    """Return the Cantor set as the substitution of every interval by its two outer parts.

    Every generation keeps the first and the last part of each interval, each `ratio` of its length, and removes the
    middle. With the ratio of one third, generation `n` has `2^n` intervals of total length `(2/3)^n`, and the limit set
    has the similarity dimension `log 2 / log 3`, about 0.631.

    Args:
        ratio:
            The length of each kept part relative to its interval, below one half.

    Returns:
        system:
            The model, with the patch `cantor` and its number of intervals `cantor.size`.

    Raises:
        ValidationError:
            The ratio is not between 0 and 1/2.

    Examples:
        >>> from skcomplex.models import run
        >>> run('cantor_set', settings={'until': 3}).trajectory_['cantor.size'].tolist()
        [1.0, 2.0, 4.0, 8.0]
    """
    if not 0 < ratio < 0.5:  # noqa: PLR2004
        error_msg = f'ratio = {ratio!r} must lie between 0 and 1/2.'
        raise ValidationError(error_msg)
    interval = Prototile('interval', [(0.0, 0.0), (1.0, 0.0)])
    pieces = [('interval', Affine(scale=ratio)), ('interval', Affine(scale=ratio, translation=(1 - ratio, 0.0)))]
    patch = Patch('cantor', prototiles=[interval], bounds=(-0.03, -0.1, 1.03, 0.1))
    return System(
        components=[patch],
        interactions=[Rewrite('rules', target='cantor', rules=[Substitution(pieces[0][0], pieces)])],
    )


@register_model(
    field='mathematics/fractals',
    tags=('fractal', 'substitution', 'rewriting system'),
    references=('Sierpiński 1915', 'Sierpiński 1916', 'Thurner, Hanel, and Klimek 2018', 'Trzaska 2023'),
    defaults={'until': 6, 'dt': 1},
)
def sierpinski_substitution(shape: str = 'triangle', *, filled: bool = True) -> System:
    """Return the Sierpiński triangle or carpet as the substitution of a polygon by scaled copies of itself.

    The triangle is replaced by its three corner copies of half its side, so generation `n` has `3^n` triangles of
    total area `(3/4)^n` times the first, and the gasket has the similarity dimension `log 3 / log 2`, about 1.585. The
    square is replaced by the eight copies of a third of its side around its middle, so generation `n` has `8^n`
    squares of area `(8/9)^n`, and the carpet has the dimension `log 8 / log 3`, about 1.893.

    Args:
        shape:
            `'triangle'` or `'carpet'`.

        filled:
            Whether the geometry visual draws the polygons as a filled image instead of outlines.

    Returns:
        system:
            The model, with the patch `sierpinski` and its number of polygons `sierpinski.size`.

    Raises:
        ValidationError:
            The shape is unknown.

    Examples:
        >>> from skcomplex.models import run
        >>> run('sierpinski_substitution', settings={'until': 2}).trajectory_['sierpinski.size'].tolist()
        [1.0, 3.0, 9.0]
    """
    if shape == 'triangle':
        height = np.sqrt(3) / 2
        prototile = Prototile('triangle', [(0.0, 0.0), (1.0, 0.0), (0.5, height)])
        corners = [(0.0, 0.0), (0.5, 0.0), (0.25, height / 2)]
        pieces = [('triangle', Affine(scale=0.5, translation=corner)) for corner in corners]
    elif shape == 'carpet':
        prototile = Prototile('square', [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        cells = [(column, row) for row in range(3) for column in range(3) if (column, row) != (1, 1)]
        pieces = [('square', Affine(scale=1 / 3, translation=(column / 3, row / 3))) for column, row in cells]
    else:
        error_msg = f'Unknown shape {shape!r}. Use one of {", ".join(SHAPES)}.'
        raise ValidationError(error_msg)
    patch = Patch('sierpinski', prototiles=[prototile], filled=filled)
    return System(
        components=[patch],
        interactions=[Rewrite('rules', target='sierpinski', rules=[Substitution(pieces[0][0], pieces)])],
    )
