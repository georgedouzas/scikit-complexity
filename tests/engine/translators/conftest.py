"""Provide the files of the system dynamics fixtures and the import of translated modules."""

import importlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from skcomplex.engine.translators import SystemDynamicsTranslator


@pytest.fixture
def fixtures_folder(request: pytest.FixtureRequest) -> Path:
    """Return the folder of the model files."""
    return request.config.rootpath / 'tests' / 'fixtures' / 'system_dynamics'


@pytest.fixture
def translate(
    fixtures_folder: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], ModuleType]:
    """Return a function that translates a fixture, saves its source, and imports it as a module."""
    monkeypatch.syspath_prepend(tmp_path)

    def load(file_name: str) -> ModuleType:
        translator = SystemDynamicsTranslator(fixtures_folder / file_name).prepare().run()
        module_name = f'translated_{file_name.replace(".", "_")}_{tmp_path.name}'
        (tmp_path / f'{module_name}.py').write_text(translator.source_)
        importlib.invalidate_caches()
        return importlib.import_module(module_name)

    return load
