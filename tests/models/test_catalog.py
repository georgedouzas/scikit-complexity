"""Test the catalog of the model library: listing, describing, creating, and running models by name."""

import ast
import importlib
import subprocess  # nosec B404: the tests start a fresh interpreter to see which modules a call loads
import sys
from pathlib import Path

import pytest

from skcomplex.core.errors import RegistryError
from skcomplex.models import catalog, create, describe, scan_listings

def _loaded(code: str) -> list[str]:
    """Return the model subfields a fresh interpreter loads to run some code."""
    probe = (
        f'{code}\n'
        'import sys\n'
        "loaded = {'.'.join(name.split('.')[:4]) for name in sys.modules if name.count('.') >= 3 "
        "and name.startswith('skcomplex.models.')}\n"
        'print(sorted(loaded))\n'
    )
    command = [sys.executable, '-c', probe]
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout  # noqa: S603  # nosec B603
    return ast.literal_eval(output)


def test_catalog_without_imports():
    """Test that listing the catalog imports no model module."""
    assert _loaded('from skcomplex.models import catalog\nassert len(catalog()) > 500') == []


def test_catalog_filters_by_field_without_imports():
    """Test that listing one field path names its models and imports none of them."""
    code = "from skcomplex.models import catalog\nassert catalog(field='physics/mechanics')"
    assert _loaded(code) == []
    assert {listing.field for listing in catalog(field='physics/mechanics')} == {'physics/mechanics'}


def test_create_imports_only_its_model():
    """Test that creating a model imports its subfield and the realizations of its field, and no other subfield."""
    loaded = _loaded("from skcomplex.models import create\ncreate('pendulum')")
    assert loaded == ['skcomplex.models.physics.mechanics', 'skcomplex.models.physics.realizations']


def test_create_unknown_name_suggests_close_names():
    """Test that an unknown model name raises an error that suggests the closest registered names."""
    with pytest.raises(RegistryError, match=r"'pendulumm'\. Did you mean pendulum"):
        create('pendulumm')


def test_scan_listings_rejects_computed_registration_argument(tmp_path: Path):
    """Test that a field path computed at run time is rejected with the module and the argument."""
    module = tmp_path / 'physics' / 'mechanics' / '_computed.py'
    module.parent.mkdir(parents=True)
    module.write_text(
        'from skcomplex.models import register_model\n\n\n'
        "@register_model(field='physics/' + 'mechanics')\n"
        'def computed():\n'
        '    """Return nothing."""\n',
    )
    message = r"skcomplex\.models\.physics\.mechanics\._computed registers the argument 'field'"
    with pytest.raises(RegistryError, match=message):
        scan_listings(tmp_path)


def test_listing_matches_registration():
    """Test that every listed model registers, once its module is imported, the metadata it was listed with."""
    for listing in catalog():
        if not listing.module.startswith('skcomplex.models.'):
            continue
        importlib.import_module(listing.module)
        entry = describe(listing.name)
        listed = (listing.name, listing.field, listing.description, listing.tags)
        assert (entry.name, entry.field, entry.description, entry.tags) == listed, listing.name


def test_engine_resolves_a_listed_model_without_its_module_imported():
    """Test that the model registry of the engine imports the module of a listed model when it resolves its name."""
    code = "import skcomplex.models\nfrom skcomplex.engine.runtime import MODELS\nMODELS.resolve('pendulum')"
    assert 'skcomplex.models.physics.mechanics' in _loaded(code)
