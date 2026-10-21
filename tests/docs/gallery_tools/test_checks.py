"""Test the checks and the encoding of screenshots."""

import numpy as np
import pytest
import rerun.blueprint as rrb
from gallery_tools import (
    GalleryError,
    encode_png,
    find_problems,
    has_bands,
    has_notification,
    is_blank,
    is_placeholder,
    trim_border,
    view_boxes,
)
from PIL import Image

SIZE = (200, 140)


def _image(color=(0, 0, 0), size=SIZE):
    """Return an image of one colour."""
    return Image.new('RGB', size, color)


def _with_lines(image):
    """Return an image with coloured lines across it, like a time series view."""
    pixels = np.asarray(image).copy()
    pixels[::10, :] = (230, 80, 60)
    return Image.fromarray(pixels)


def test_view_boxes_split_containers():
    """Test that the boxes follow the shares of a vertical layout and the cells of a grid."""
    layout = rrb.Vertical(
        rrb.Spatial2DView(origin='/a', name='a'),
        rrb.Grid(*[rrb.TimeSeriesView(origin=f'/{name}', name=name) for name in 'bcd'], grid_columns=2),
        row_shares=[1, 1],
    )
    assert view_boxes(layout, (100, 100)) == [
        ('a', (0, 0, 100, 50)),
        ('b', (0, 50, 50, 75)),
        ('c', (50, 50, 100, 75)),
        ('d', (0, 75, 50, 100)),
    ]


def test_is_blank_detects_uniform_and_grey_views():
    """Test that a single colour and the grey chrome of the viewer are blank, and coloured content is not."""
    assert is_blank(np.asarray(_image()))
    grey = np.asarray(_image((40, 40, 40))).copy()
    grey[::7] = (120, 120, 120)
    assert is_blank(grey)
    assert not is_blank(np.asarray(_with_lines(_image())))


def test_is_blank_accepts_single_channel_image():
    """Test that a greyscale image of dark and light grey cells, as a single-channel `rr.Image` shows, is not blank."""
    cells = np.kron(np.random.default_rng(0).integers(0, 2, (7, 10)), np.ones((20, 20), dtype=int)) * 140 + 60
    pixels = np.repeat(cells[..., None], 3, axis=-1).astype(np.uint8)
    assert not is_blank(pixels)
    assert find_problems(Image.fromarray(pixels), SIZE, [('pattern', (0, 0, *SIZE))]) == []


def test_has_notification_detects_toast():
    """Test that a notification box in the top right is detected."""
    pixels = np.asarray(_image()).copy()
    pixels[5:30, 120:195] = (33, 33, 33)
    assert has_notification(pixels)
    assert not has_notification(np.asarray(_image()))


def test_find_problems_names_blank_view_and_size():
    """Test that a blank view and a wrong size are reported."""
    image = Image.fromarray(
        np.concatenate(
            [np.asarray(_with_lines(_image((0, 0, 0), (100, 140)))), np.zeros((140, 100, 3), np.uint8)],
            axis=1,
        ),
    )
    boxes = view_boxes(rrb.Horizontal(rrb.TimeSeriesView(name='lines'), rrb.Spatial2DView(name='field')), SIZE)
    assert find_problems(image, SIZE, boxes) == ['the view field is blank']
    assert find_problems(image, (400, 280), boxes) == ['the size is 200x140, not 400x280']


def test_find_problems_names_notification():
    """Test that a notification over a view is reported."""
    pixels = np.asarray(_with_lines(_image())).copy()
    pixels[5:30, 110:195] = (33, 33, 33)
    problems = find_problems(Image.fromarray(pixels), SIZE, [('view', (0, 0, *SIZE))])
    assert problems == ['a notification covers a view']


def test_encode_png_keeps_size_within_limit(tmp_path):
    """Test that an image larger than the limit is reduced to a palette."""
    noise = np.random.default_rng(0).integers(0, 4, (280, 400, 3), dtype=np.uint8) * 60
    path = encode_png(Image.fromarray(noise), tmp_path / 'noise.png', 100_000)
    with Image.open(path) as image:
        assert image.format == 'PNG'
        assert image.size == (400, 280)
    assert path.stat().st_size <= 100_000


def test_encode_png_raises_over_limit(tmp_path):
    """Test that an image that stays over the limit is rejected."""
    noise = np.random.default_rng(0).integers(0, 255, (280, 400, 3), dtype=np.uint8)
    with pytest.raises(GalleryError, match='more than 1000'):
        encode_png(Image.fromarray(noise), tmp_path / 'noise.png', 1000)


def test_is_placeholder_compares_rescaled_images(tmp_path):
    """Test that a rescaled copy of the placeholder is recognised and a screenshot is not."""
    placeholder = tmp_path / 'no_image.png'
    _with_lines(_image((255, 255, 255), (400, 280))).save(placeholder)
    _with_lines(_image((255, 255, 255), (400, 280))).resize(SIZE).save(tmp_path / 'thumb.png')
    _image((0, 0, 0)).save(tmp_path / 'screenshot.png')
    assert is_placeholder(tmp_path / 'thumb.png', placeholder)
    assert not is_placeholder(tmp_path / 'screenshot.png', placeholder)


def test_find_problems_names_light_chrome():
    """Test that the panels of the light theme, as a persisted viewer state draws them, are reported."""
    pixels = np.asarray(_with_lines(_image())).copy()
    pixels[:, :80] = (240, 240, 240)
    problems = find_problems(Image.fromarray(pixels), SIZE, [('view', (0, 0, *SIZE))])
    assert problems[0] == 'the viewer shows its light theme or its panels'


def test_find_problems_names_letterbox_bands():
    """Test that wide uniform bands on two sides of a view are reported, and trimmed away."""
    pixels = np.zeros((140, 200, 3), dtype=np.uint8)
    pixels[:, 30:170] = np.asarray(_with_lines(_image((20, 20, 120), (140, 140))))
    problems = find_problems(Image.fromarray(pixels), SIZE, [('view', (0, 0, *SIZE))])
    assert problems == ['a uniform band covers more than a tenth of the image along an edge']
    assert trim_border(Image.fromarray(pixels)).size == (140, 140)


def test_has_notification_ignores_grey_content():
    """Test that grey content of the notification colour, as a thin strip, is not a notification."""
    pixels = np.asarray(_image()).copy()
    pixels[:, 150:156] = (33, 33, 33)
    assert not has_notification(pixels)


def test_find_problems_accepts_sparse_content():
    """Test that points with empty margins leave no band, and are not trimmed."""
    pixels = np.zeros((140, 200, 3), dtype=np.uint8)
    pixels[0:6, 0:6] = pixels[134:140, 194:200] = pixels[60:66, 100:106] = (230, 80, 60)
    assert not has_bands(pixels)
    assert find_problems(Image.fromarray(pixels), SIZE, [('view', (0, 0, *SIZE))]) == []
    assert trim_border(Image.fromarray(pixels)).size == SIZE


def test_find_problems_accepts_coloured_background():
    """Test that a lattice whose background colour fills the edges is content, not letterboxing."""
    pixels = np.zeros((140, 200, 3), dtype=np.uint8)
    pixels[:] = (68, 1, 84)
    pixels[:, 90:110] = (33, 145, 140)
    assert not has_bands(pixels)


def test_encode_png_uses_adaptive_palette(tmp_path):
    """Test that an image over the limit is written with a palette."""
    noise = np.random.default_rng(0).integers(0, 4, (280, 400, 3), dtype=np.uint8) * 60
    path = encode_png(Image.fromarray(noise), tmp_path / 'noise.png', 100_000)
    with Image.open(path) as image:
        assert image.mode == 'P'
