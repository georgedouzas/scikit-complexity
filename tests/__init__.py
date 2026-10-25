"""Tests suite for `skcomplex`."""

from importlib.resources import files
from pathlib import Path

TESTS_DIR = Path(str(files('tests')))
TMP_DIR = TESTS_DIR / 'tmp'
FIXTURES_DIR = TESTS_DIR / 'fixtures'
