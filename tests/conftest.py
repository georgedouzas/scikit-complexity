"""Configuration for the pytest test suite."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the option that runs the timing benchmarks, which need an otherwise idle machine."""
    parser.addoption('--benchmarks', action='store_true', help='Run the timing benchmarks.')


def pytest_configure(config: pytest.Config) -> None:
    """Register the benchmark marker."""
    config.addinivalue_line('markers', 'benchmark: a timing benchmark, run with --benchmarks on an idle machine')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the timing benchmarks unless they are requested."""
    if config.getoption('--benchmarks'):
        return
    skip = pytest.mark.skip(reason='timing benchmark, run with --benchmarks on an idle machine')
    for item in items:
        if 'benchmark' in item.keywords:
            item.add_marker(skip)
