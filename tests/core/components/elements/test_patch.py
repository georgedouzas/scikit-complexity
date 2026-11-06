"""Test the `_patch` module of `skcomplex.core.components.elements`."""

import pytest

from skcomplex.core.components.elements import PatchElement
from skcomplex.core.errors import RewritingError
from skcomplex.core.specs import AffineSpec, PrototileSpec

INTERVAL = PrototileSpec('interval', [(0, 0), (1, 0)])


@pytest.mark.parametrize(
    ('patch', 'message'),
    [
        (PatchElement('cantor'), 'at least one prototile'),
        (PatchElement('cantor', [('square', AffineSpec())], prototiles=[INTERVAL]), "unknown prototile 'square'"),
        (PatchElement('cantor', [('interval', AffineSpec(translation=(0,)))], prototiles=[INTERVAL]), 'translation'),
    ],
)
def test_patch_checks_its_prototiles_and_tiles(patch, message):
    """Test that a patch needs prototiles and tiles of known prototiles at valid placements."""
    with pytest.raises(RewritingError, match=message):
        patch.prepare()


def test_patch_is_drawn_as_its_geometry():
    """Test that a prepared patch is drawn as the geometry of its tiles, as outlines by default."""
    patch = PatchElement('cantor', [('interval', AffineSpec())], prototiles=[INTERVAL]).prepare()
    assert (patch.default_visual_.kind, patch.filled) == ('geometry', False)
