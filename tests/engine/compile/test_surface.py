"""Test that the public names of the compile package are its classes, bases, and pipeline functions."""

import inspect

from skcomplex.engine import compile as package

FUNCTIONS = {
    'claiming_compiler',
    'compile_system',
    'compiled_problem',
    'observables_of',
    'order_passes',
    'partition_system',
    'run_passes',
    'validate',
}


def test_public_names_are_classes_or_pipeline_functions():
    """Every public name is a class or one of the pipeline functions."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)) or name in FUNCTIONS, name


def test_pipeline_functions_are_public():
    """Every pipeline function is public and callable."""
    for name in FUNCTIONS:
        assert name in package.__all__
        assert callable(getattr(package, name))


def test_helpers_of_one_pass_are_not_public():
    """The helpers of a single pass are not public."""
    assert not {'values_of', 'member_of', 'hilbert_of'} & set(package.__all__)
