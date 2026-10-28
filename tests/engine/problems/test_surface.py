"""Test that the public names of the problems package are its classes, values, and the functions its clients use."""

import inspect

from skcomplex.engine import problems as package

FUNCTIONS = {
    'components_of',
    'contains',
    'declarations_of',
    'declared_variables',
    'decode',
    'default_of',
    'domain_kind',
    'domain_of',
    'encode',
    'labels_of',
    'numpy_function',
    'print_expressions',
    'step_count',
    'symbol',
    'symbols_of',
    'vector',
    'wrap_angle',
}
VALUES = {'Getter', 'LATTICE_BOUNDARIES', 'MODES', 'NUMPY_DIALECT', 'SCALAR_DIALECT', 'VectorFunction'}


def test_public_names_are_classes_values_or_functions():
    """Every public name is a class, a value a client reads, or one of the kept functions."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)) or name in FUNCTIONS | VALUES, name


def test_kept_functions_are_public():
    """Every kept function is public and callable."""
    for name in FUNCTIONS:
        assert name in package.__all__
        assert callable(getattr(package, name))
