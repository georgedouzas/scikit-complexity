"""Hold the settings of the gallery: image sizes, waits, encoding, and the recording budget."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from dataclasses import dataclass

BACKENDS = ('native', 'web')
DEFAULT_BACKEND = 'native'


@dataclass(frozen=True)
class CaptureSettings:
    """Settings of a screenshot capture, which are part of the cache key.

    Args:
        thumbnail_size:
            The size in pixels of the thumbnail, at the 10:7 ratio of the gallery's `thumbnail_size`.

        density:
            The number of pixels per logical point, so the viewer lays out views at the size divided by it.

        wait:
            The seconds to wait for a recording to load before the first capture, doubled for the retry.

        settle:
            The seconds the viewer state or the rendered image must stay unchanged to count as loaded.

        max_png_bytes:
            The largest size of an encoded screenshot.

        version:
            The version of the capture procedure, raised whenever the capture or its processing changes the images.
    """

    thumbnail_size: tuple[int, int] = (1200, 840)
    density: int = 2
    wait: float = 20.0
    settle: float = 0.4
    max_png_bytes: int = 300_000
    version: int = 4

    def logical(self, size: tuple[int, int]) -> tuple[int, int]:
        """Return the logical size of an image size.

        Args:
            size:
                The size in pixels.

        Returns:
            size:
                The size in logical points.

        Examples:
            >>> from gallery_tools import CaptureSettings
            >>> CaptureSettings().logical((1200, 840))
            (600, 420)
        """
        return size[0] // self.density, size[1] // self.density


@dataclass(frozen=True)
class Budget:
    """Limits every example runs within, so the capture loads the whole recording.

    Args:
        max_frames:
            The largest number of frames on the timeline of a recording.

        max_bytes:
            The largest size of a recording file.

        max_seconds:
            The longest run time of an example.
    """

    max_frames: int = 300
    max_bytes: int = 20_000_000
    max_seconds: float = 60.0
