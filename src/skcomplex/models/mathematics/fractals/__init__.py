"""Fractals: escape-time sets of the complex quadratic map."""

from ._quadratic import QuadraticMap, julia, mandelbrot

__all__: list[str] = [
    'QuadraticMap',
    'julia',
    'mandelbrot',
]
