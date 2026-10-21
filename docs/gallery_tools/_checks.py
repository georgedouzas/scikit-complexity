"""Check screenshots with image statistics and encode them as optimised PNG files."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rerun.blueprint as rrb
from PIL import Image

from ._errors import GalleryError

type Box = tuple[int, int, int, int]

MIN_DEVIATION = 2.0
MIN_COLORFUL_SHARE = 0.0005
MIN_SATURATION = 0.25
MIN_BRIGHTNESS = 40
MIN_CONTRAST = 32
BLOCK = 4
MIN_AREA_SHARE = 0.05
NOTIFICATION_COLOR = (33, 33, 33)
NOTIFICATION_TOLERANCE = 2
MAX_NOTIFICATION_SHARE = 0.05
MIN_NOTIFICATION_WIDTH = 0.25
MIN_NOTIFICATION_FILL = 0.5
MARGIN = 0.02
PLACEHOLDER_TOLERANCE = 2.0
LIGHT_GREY = 215
MAX_LIGHT_SHARE = 0.2
BORDER_TOLERANCE = 8
MAX_BAND_SHARE = 0.1
MIN_SOLID_SHARE = 0.5
PALETTE_SIZES = (256, 128, 64, 32, 16)
QUOTED = re.compile(r'`([A-Za-z_][\w./]*)`')
SENTENCE_END = re.compile(r'(?<=\.)\s+')


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


def _area_share(pixels: npt.NDArray[np.uint8]) -> float:
    """Return the share of blocks of pixels whose brightness all differ from the commonest brightness.

    The lines and the labels of the viewer chrome are thinner than a block, while the cells of a greyscale image fill
    whole blocks.
    """
    brightness = pixels.astype(int).mean(axis=-1).astype(int)
    differs = np.abs(brightness - np.bincount(brightness.ravel()).argmax()) > MIN_CONTRAST
    rows, columns = differs.shape[0] // BLOCK, differs.shape[1] // BLOCK
    if not rows or not columns:
        return 0.0
    blocks = differs[: rows * BLOCK, : columns * BLOCK].reshape(rows, BLOCK, columns, BLOCK)
    return float(blocks.all(axis=(1, 3)).mean())


def is_blank(pixels: npt.NDArray[np.uint8]) -> bool:
    """Return whether a view shows nothing: a single colour, or only the grey of the viewer chrome.

    A view shows content when a share of its pixels has a saturated colour, or when areas of it differ in brightness
    from the commonest brightness, as the cells of a single-channel image do.

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
    return _colorful_share(pixels) < MIN_COLORFUL_SHARE and _area_share(pixels) < MIN_AREA_SHARE


def has_notification(pixels: npt.NDArray[np.uint8]) -> bool:
    """Return whether a viewer notification covers the top right of an image, by its background colour.

    A notification is a filled panel of its background colour, at least a quarter as wide as the top right quarter of
    the image. Content of the same grey, such as a thin strip or scattered pixels, fills no such panel.

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
        >>> pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> pixels[2:20, 60:98] = 33
        >>> has_notification(pixels)
        True
    """
    height, width = pixels.shape[:2]
    corner = pixels[: height // 4, width // 2 :].astype(int)
    matches = np.all(np.abs(corner - np.array(NOTIFICATION_COLOR)) <= NOTIFICATION_TOLERANCE, axis=-1)
    if not matches.size or float(matches.mean()) <= MAX_NOTIFICATION_SHARE:
        return False
    rows, columns = np.flatnonzero(matches.any(axis=1)), np.flatnonzero(matches.any(axis=0))
    panel = matches[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    wide = panel.shape[1] >= MIN_NOTIFICATION_WIDTH * matches.shape[1]
    return wide and float(panel.mean()) >= MIN_NOTIFICATION_FILL


def _content_box(pixels: npt.NDArray[np.uint8]) -> Box | None:
    """Return the box of the pixels that differ from the top left corner, or `None` when none does."""
    differs = np.any(np.abs(pixels.astype(int) - pixels[0, 0].astype(int)) > BORDER_TOLERANCE, axis=-1)
    rows, columns = np.flatnonzero(differs.any(axis=1)), np.flatnonzero(differs.any(axis=0))
    if not rows.size:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def _letterbox(pixels: npt.NDArray[np.uint8]) -> Box | None:
    """Return the box of an image a view draws inside bands of its background, or `None` for other content.

    The content is such an image when the corner has the grey of the view background, its box spans the width or the
    height of the view, as a view fits an image, and the edges of its box are mostly not of the background colour.
    Sparse content, such as points or a network, leaves background along the edges of its box, and is not letterboxed.
    """
    box = _content_box(pixels)
    if box is None or _colorful_share(pixels[:1, :1]) > 0:
        return None
    height, width = pixels.shape[:2]
    left, top, right, bottom = box
    tall = top <= MARGIN * height and bottom >= (1 - MARGIN) * height
    wide = left <= MARGIN * width and right >= (1 - MARGIN) * width
    if not tall and not wide:
        return None
    inside = pixels[top:bottom, left:right].astype(int)
    edges = np.concatenate([inside[0], inside[-1], inside[:, 0], inside[:, -1]])
    differs = np.any(np.abs(edges - pixels[0, 0].astype(int)) > BORDER_TOLERANCE, axis=-1)
    return box if float(differs.mean()) > MIN_SOLID_SHARE else None


def trim_border(image: Image.Image) -> Image.Image:
    """Return an image without the bands of background a view leaves around an image of a fixed aspect.

    A view with a fixed aspect, such as a square lattice, leaves bands of background on two sides of a wider view.
    Sparse content, such as points or a network, is left as it is, since its empty margins belong to the view.

    Args:
        image:
            The image.

    Returns:
        image:
            The image of the view without its bands, or the image itself when the view draws no letterboxed image.

    Examples:
        >>> from PIL import Image
        >>> from gallery_tools import trim_border
        >>> image = Image.new('RGB', (100, 70))
        >>> image.paste((200, 50, 50), (30, 0, 70, 70))
        >>> trim_border(image).size
        (40, 70)
        >>> image = Image.new('RGB', (100, 70))
        >>> image.paste((200, 50, 50), (30, 0, 34, 4))
        >>> image.paste((200, 50, 50), (60, 66, 64, 70))
        >>> trim_border(image).size
        (100, 70)
    """
    image = image.convert('RGB')
    box = _letterbox(np.asarray(image))
    return image.crop(box) if box else image


def has_bands(pixels: npt.NDArray[np.uint8]) -> bool:
    """Return whether an image is letterboxed by bands wider than a tenth of its side, as a view with a fixed aspect is.

    Args:
        pixels:
            The RGB pixels of the image.

    Returns:
        bands:
            Whether an image drawn inside bands of the colour of the top left corner leaves a band wider than a tenth
            of the width or the height. Sparse content with empty margins has no bands.

    Examples:
        >>> import numpy as np
        >>> from gallery_tools import has_bands
        >>> pixels = np.zeros((70, 100, 3), dtype=np.uint8)
        >>> pixels[:, 20:80] = (200, 50, 50)
        >>> has_bands(pixels)
        True
        >>> pixels = np.zeros((70, 100, 3), dtype=np.uint8)
        >>> pixels[0:4, 40:44] = pixels[66:70, 60:64] = (200, 50, 50)
        >>> has_bands(pixels)
        False
    """
    box = _letterbox(pixels)
    if box is None:
        return False
    height, width = pixels.shape[:2]
    left, top, right, bottom = box
    return max(top, height - bottom) > MAX_BAND_SHARE * height or max(left, width - right) > MAX_BAND_SHARE * width


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
    """Write an image as an optimised PNG, reduced to smaller adaptive palettes while it is larger than the limit.

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
        palette = image.convert('P', palette=Image.Palette.ADAPTIVE, colors=colors, dither=Image.Dither.NONE)
        palette.save(path, format='PNG', optimize=True)
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


def caption_series(source: str) -> list[str]:
    """Return the names an example's comments quote where they describe the series of its primary view.

    A sentence of the comments that names the primary view and series is read, and every identifier or entity path it
    quotes in backticks is taken as a series the primary view shows.

    Args:
        source:
            The source of the example.

    Returns:
        names:
            The quoted names, in the order of the comments.

    Examples:
        >>> from gallery_tools import caption_series
        >>> caption_series('# Its primary view is the time series of `q` and `m`.')
        ['q', 'm']
    """
    lines = [line.strip()[1:].strip() for line in source.splitlines() if line.strip().startswith('#')]
    sentences = SENTENCE_END.split(' '.join(line for line in lines if not line.startswith('%%')))
    names: list[str] = []
    for sentence in sentences:
        if 'primary view' in sentence and 'series' in sentence:
            names += [name for name in QUOTED.findall(sentence) if name not in names]
    return names


def _expressions(view: rrb.View) -> list[tuple[bool, str]]:
    """Return the entity path expressions of a view's contents, whether each includes, and its path pattern."""
    contents = view.contents if view.contents is not None else '$origin/**'
    lines = contents.splitlines() if isinstance(contents, str) else [str(line) for line in contents]
    origin = str(view.origin).rstrip('/')
    expressions = []
    for line in (line.strip().replace('$origin', origin) for line in lines if line.strip()):
        include = not line.startswith('-')
        expressions.append((include, line.lstrip('+-').strip()))
    return expressions


def _matches(path: str, pattern: str) -> bool:
    """Return whether an entity path matches a pattern, a path or a path followed by `/**`."""
    if pattern.endswith('/**'):
        base = pattern[:-3].rstrip('/')
        return path == base or path.startswith(f'{base}/')
    return path == pattern


def missing_series(names: Sequence[str], paths: Sequence[str], view: rrb.View) -> list[str]:
    """Return the names of series that no entity path a view includes ends with.

    Args:
        names:
            The names of the series, as `caption_series` reads them.

        paths:
            The entity paths of a recording.

        view:
            The view, whose contents are paths and `/**` subtrees, included or, after a leading `-`, excluded.

    Returns:
        names:
            The names the view shows no series of.

    Examples:
        >>> import rerun.blueprint as rrb
        >>> from gallery_tools import missing_series
        >>> missing_series(['x', 'y'], ['/lorenz/x', '/lorenz/y'], rrb.TimeSeriesView(origin='/', contents='/lorenz/x'))
        ['y']
    """
    expressions = _expressions(view)
    included = []
    for path in paths:
        shown = False
        for include, pattern in expressions:
            if _matches(path, pattern):
                shown = include
        if shown:
            included.append(path)
    return [name for name in names if not any(path.endswith(f'/{name.strip("/")}') for path in included)]
