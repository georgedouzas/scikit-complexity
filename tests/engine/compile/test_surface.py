"""Test that the public names of the compile package are its bases, its compilers, its values, and `compile_system`."""

import inspect

from skcomplex.engine import compile as package

FUNCTIONS = {'compile_system'}


def test_public_names_are_classes_or_compile_system():
    """Every public name is a class, or `compile_system`."""
    for name in package.__all__:
        assert inspect.isclass(getattr(package, name)) or name in FUNCTIONS, name


def test_compilers_are_public():
    """The bases and the compilers are public, and every compiler derives from the compiler base."""
    for name in ('BaseCompiler', 'BasePass', 'BaseLowering', 'PassCompiler', 'QuantumCompiler', 'RewritingCompiler'):
        assert name in package.__all__
    for compiler in (package.PassCompiler, package.QuantumCompiler, package.RewritingCompiler):
        assert issubclass(compiler, package.BaseCompiler)


def test_removed_functions_are_not_public():
    """The functions that became methods of the compiler are not public."""
    removed = {'validate', 'claiming_compiler', 'compiled_problem', 'observables_of', 'order_passes', 'run_passes'}
    assert not (removed | {'partition_system'}) & set(package.__all__)


def test_helpers_of_one_pass_are_not_public():
    """The helpers of a single pass are not public."""
    assert not {'values_of', 'member_of', 'hilbert_of'} & set(package.__all__)
