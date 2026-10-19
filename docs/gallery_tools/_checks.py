"""Check screenshots with image statistics and encode them as optimised PNG files."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rerun.blueprint as rrb
from PIL import Image, ImageChops

from ._errors import GalleryError

type Box = tuple[int, int, int, int]

MIN_DEVIATION = 2.0
MIN_COLORFUL_SHARE = 0.0005
MIN_SATURATION = 0.25
MIN_BRIGHTNESS = 40
NOTIFICATION_COLOR = (33, 33, 33)
NOTIFICATION_TOLERANCE = 2
MAX_NOTIFICATION_SHARE = 0.05
MARGIN = 0.02
PLACEHOLDER_TOLERANCE = 2.0
LIGHT_GREY = 215
MAX_LIGHT_SHARE = 0.2
BORDER_TOLERANCE = 8
MAX_BAND_SHARE = 0.1
PALETTE_SIZES = (256, 128, 64, 32)


def _split(box: Box, shares: Sequence[float], *, horizontal: bool) -> list[Box]:
    """Return the parts of a box split by shares along one axis."""
    left, top, right, bottom = box
    total = float(sum(shares))
    edges = np.cumsum([0.0, *shares]) / total
    if horizontal:
        positions = [round(left + edge * (right - left)) for edge in edges]
        return [(start, top, end, bottom) for start, end in itertools.pairwise(positions)]
    positions = [round(top + edge * (bottom - top)) for edge in edges]
    return [(left, start, right, end) for start, end in itertools.pairwise(positions)]


def _grid(box: Box, count: int, columns: int | None) -> list[Box]:
    """Return the cells of a grid of a number of parts, filled row by row."""
    columns = columns or max(1, round(np.ceil(np.sqrt(count))))
    rows = int(np.ceil(count / columns))
    cells = []
    for row in _split(box, [1.0] * rows, horizontal=False):
        cells += _split(row, [1.0] * columns, horizontal=True)
    return cells[:count]


def _parts(container: rrb.Container, box: Box) -> list[tuple[rrb.View | rrb.Container, Box]]:
    """Return the children of a container with the box each one fills."""
    children = list(container.contents)
    if isinstance(container, rrb.Tabs):
        return [(children[0], box)] if children else []
    if isinstance(container, rrb.Grid):
        return list(zip(children, _grid(box, len(children), container.grid_columns), strict=True))
    horizontal = isinstance(container, rrb.Horizontal)
    given = container.column_shares if horizontal else container.row_shares
    shares = [float(share) for share in given] if isinstance(given, list | tuple) else [1.0] * len(children)
    return list(zip(children, _split(box, shares, horizontal=horizontal), strict=True))


def view_boxes(part: rrb.View | rrb.Container, size: tuple[int, int]) -> list[tuple[str, Box]]:
    """Return the name and the box of every view of a layout drawn at a size.

    Args:
        part:
            The root container or view of a blueprint.

        size:
            The size of the image the layout fills.

    Returns:
        boxes:
            The name of each view, or its origin, with its box as left, top, right, and bottom pixels.

    Examples:
        >>> import rerun.blueprint as rrb
        >>> from gallery_tools import view_boxes
        >>> layout = rrb.Horizontal(rrb.Spatial2DView(origin='/u', name='u'), rrb.Spatial2DView(origin='/v', name='v'))
        >>> view_boxes(layout, (200, 100))
        [('u', (0, 0, 100, 100)), ('v', (100, 0, 200, 100))]
    """
    stack: list[tuple[rrb.View | rrb.Container, Box]] = [(part, (0, 0, *size))]
    boxes = []
    while stack:
        item, box = stack.pop(0)
        if isinstance(item, rrb.View):
            boxes.append((str(item.name or item.origin), box))
        else:
            stack = _parts(item, box) + stack
    return boxes


def _inner(box: Box, title: int) -> Box:
    """Return the inside of a view box, without its title bar and its edges."""
    left, top, right, bottom = box
    top += title
    dx, dy = round((right - left) * MARGIN), round((bottom - top) * MARGIN)
    return left + dx, top + dy, right - dx, bottom - dy


def _colorful_share(pixels: npt.NDArray[np.uint8]) -> float:
    """Return the share of pixels with a saturated, visible colour, which the viewer chrome never has."""
    values = pixels.astype(float)
    high, low = values.max(axis=-1), values.min(axis=-1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1.0), 0.0)
    return float(np.mean((saturation > MIN_SATURATION) & (high > MIN_BRIGHTNESS)))


def is_blank(pixels: npt.NDArray[np.uint8]) -> bool:
    """Return whether a view shows nothing: a single colour, or only the grey of the viewer chrome.

    Args:
        pixels:
            The RGB pixels of the inside of a view.

    Returns:
        blank:
            Whether the view is blank.

    Examples:
        >>> import numpy as np
        >>> from gallery_tools import is_blank
        >>> is_blank(np.zeros((10, 10, 3), dtype=np.uint8))
        True
    """
    if not pixels.size or float(pixels.std()) < MIN_DEVIATION:
        return True
    return _colorful_share(pixels) < MIN_COLORFUL_SHARE


def has_notification(pixels: npt.NDArray[np.uint8]) -> bool:
    """Return whether a viewer notification covers the top right of an image, by its background colour.

    Args:
        pixels:
            The RGB pixels of the image, without the title bars of its views.

    Returns:
        notification:
            Whether a notification is drawn.

    Examples:
        >>> import numpy as np
        >>> from gallery_tools import has_notification
        >>> has_notification(np.zeros((100, 100, 3), dtype=np.uint8))
        False
    """
    height, width = pixels.shape[:2]
    corner = pixels[: height // 4, width // 2 :].astype(int)
    matches = np.all(np.abs(corner - np.array(NOTIFICATION_COLOR)) <= NOTIFICATION_TOLERANCE, axis=-1)
    return bool(matches.size) and float(matches.mean()) > MAX_NOTIFICATION_SHARE


def trim_border(image: Image.Image) -> Image.Image:
    """Return an image without the bands of the colour of its top left corner along its edges.

    A view with a fixed aspect, such as a square lattice, leaves bands of background on two sides of a wider view.

    Args:
        image:
            The image.

    Returns:
        image:
            The smallest part of the image that holds everything not of the corner colour.

    Examples:
        >>> from PIL import Image
        >>> from gallery_tools import trim_border
        >>> image = Image.new('RGB', (100, 70))
        >>> image.paste((200, 50, 50), (30, 0, 70, 70))
        >>> trim_border(image).size
        (40, 70)
    """
    image = image.convert('RGB')
    corner = image.getpixel((0, 0))
    difference = ImageChops.difference(image, Image.new('RGB', image.size, corner)).convert('L')
    box = difference.point(lambda value: 255 if value > BORDER_TOLERANCE else 0).getbbox()
    return image.crop(box) if box else image


def _band(pixels: npt.NDArray[np.uint8]) -> int:
    """Return the number of leading rows of an image that hold one colour, the colour of its first row."""
    values = pixels.astype(int)
    uniform = np.all(np.abs(values - values[0, 0]) <= BORDER_TOLERANCE, axis=(1, 2))
    return int(np.argmin(uniform)) if not uniform.all() else len(uniform)


def has_bands(pixels: npt.NDArray[np.uint8]) -> bool:
    """Return whether an image has a uniform band along an edge wider than a tenth of its side, as letterboxing leaves.

    Args:
        pixels:
            The RGB pixels of the image.

    Returns:
        bands:
            Whether a band of one colour covers more than a tenth of the width or the height from an edge, in an
            image that is not of one colour throughout.

    Examples:
        >>> import numpy as np
        >>> from gallery_tools import has_bands
        >>> pixels = np.zeros((70, 100, 3), dtype=np.uint8)
        >>> pixels[:, 20:80] = (200, 50, 50)
        >>> has_bands(pixels)
        True
    """
    height, width = pixels.shape[:2]
    rows = [_band(pixels), _band(pixels[::-1])]
    columns = [_band(pixels.transpose(1, 0, 2)), _band(pixels.transpose(1, 0, 2)[::-1])]
    if max(rows) == height:
        return False
    return max(rows) > MAX_BAND_SHARE * height or max(columns) > MAX_BAND_SHARE * width


def has_light_chrome(pixels: npt.NDArray[np.uint8]) -> bool:
    """Return whether an image shows the light theme of the viewer, whose panels are near-white grey.

    The captures use the dark theme, so a large share of near-white grey shows panels or a persisted light theme.

    Args:
        pixels:
            The RGB pixels of the image.

    Returns:
        light:
            Whether more than a fifth of the pixels are near-white grey.

    Examples:
        >>> import numpy as np
        >>> from gallery_tools import has_light_chrome
        >>> has_light_chrome(np.full((10, 10, 3), 240, dtype=np.uint8))
        True
    """
    values = pixels.astype(int)
    grey = (values.max(axis=-1) - values.min(axis=-1)) < MIN_DEVIATION * 4
    return float(np.mean(grey & (values.min(axis=-1) > LIGHT_GREY))) > MAX_LIGHT_SHARE


def find_problems(
    image: Image.Image,
    size: tuple[int, int],
    boxes: Sequence[tuple[str, Box]],
    title: int = 0,
) -> list[str]:
    """Return what is wrong with a screenshot: its size, blank views, notifications, and the viewer chrome.

    Args:
        image:
            The screenshot.

        size:
            The expected size in pixels.

        boxes:
            The name and box of every view.

        title:
            The height in pixels of the title bar of each view.

    Returns:
        problems:
            One sentence per failed check, empty when the screenshot passes.

    Examples:
        >>> from PIL import Image, ImageChops
        >>> from gallery_tools import find_problems
        >>> find_problems(Image.new('RGB', (40, 28)), (40, 28), [('v', (0, 0, 40, 28))])
        ['the view v is blank']
    """
    if image.size != size:
        return [f'the size is {image.size[0]}x{image.size[1]}, not {size[0]}x{size[1]}']
    pixels = np.asarray(image.convert('RGB'))
    problems = []
    for name, box in boxes:
        left, top, right, bottom = _inner(box, title)
        if is_blank(pixels[top:bottom, left:right]):
            problems.append(f'the view {name} is blank')
    if any(has_notification(pixels[box[1] + title : box[3], box[0] : box[2]]) for _, box in boxes):
        problems.append('a notification covers a view')
    if has_light_chrome(pixels):
        problems.append('the viewer shows its light theme or its panels')
    if has_bands(pixels):
        problems.append('a uniform band covers more than a tenth of the image along an edge')
    return problems


def encode_png(image: Image.Image, path: Path, max_bytes: int) -> Path:
    """Write an image as an optimised PNG, reduced to smaller and smaller palettes while it is larger than the limit.

    Args:
        image:
            The image.

        path:
            The path of the PNG file.

        max_bytes:
            The largest size of the file.

    Returns:
        path:
            The path of the written file.

    Raises:
        GalleryError:
            The image is larger than the limit even with a palette.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from PIL import Image, ImageChops
        >>> from gallery_tools import encode_png
        >>> path = encode_png(Image.new('RGB', (120, 84)), Path(tempfile.mkdtemp()) / 'blank.png', 300_000)
        >>> path.stat().st_size < 300_000
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert('RGB')
    image.save(path, format='PNG', optimize=True)
    if path.stat().st_size <= max_bytes:
        return path
    for colors in PALETTE_SIZES:
        image.quantize(colors=colors, dither=Image.Dither.NONE).save(path, format='PNG', optimize=True)
        if path.stat().st_size <= max_bytes:
            return path
    error_msg = (
        f'{path.name} takes {path.stat().st_size} bytes with a palette, more than {max_bytes}. Simplify the view.'
    )
    raise GalleryError(error_msg)


def is_placeholder(path: Path, placeholder: Path) -> bool:
    """Return whether a thumbnail is the gallery's placeholder image, rescaled or not.

    Args:
        path:
            The thumbnail.

        placeholder:
            The placeholder image of `mkdocs-gallery`.

    Returns:
        placeholder:
            Whether the thumbnail shows the placeholder.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from PIL import Image, ImageChops
        >>> from gallery_tools import is_placeholder
        >>> folder = Path(tempfile.mkdtemp())
        >>> Image.new('RGB', (40, 28), 'white').save(folder / 'a.png')
        >>> Image.new('RGB', (80, 56), 'white').save(folder / 'b.png')
        >>> is_placeholder(folder / 'a.png', folder / 'b.png')
        True
    """
    with Image.open(path) as thumbnail, Image.open(placeholder) as reference:
        shown = np.asarray(thumbnail.convert('RGB'), dtype=float)
        expected = np.asarray(reference.convert('RGB').resize(thumbnail.size), dtype=float)
    return float(np.mean(np.abs(shown - expected))) < PLACEHOLDER_TOLERANCE
