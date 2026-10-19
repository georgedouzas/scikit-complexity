"""Capture screenshots of Rerun recordings with the headless native viewer or the web viewer in Chromium."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import copy
import functools
import io
import os
import shutil
import signal
import socket
import subprocess  # nosec B404: the capture runs the Rerun viewer installed with the interpreter
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import rerun as rr
import rerun.blueprint as rrb
from PIL import Image, ImageOps
from rerun.experimental import ViewerClient, ViewerState

from ._checks import encode_png, find_problems, trim_border, view_boxes
from ._errors import GalleryError
from ._settings import BACKENDS, CaptureSettings

POLL_SECONDS = 0.05
STABLE_POLLS = 4
STARTUP_SECONDS = 20.0
TIMELINE = 'sim_time'
NANOSECONDS = 1e9
HIDDEN_PANELS = (
    rrb.TimePanel(state='hidden', timeline=TIMELINE, play_state='following'),
    rrb.TopPanel(state='hidden'),
    rrb.BlueprintPanel(state='hidden'),
    rrb.SelectionPanel(state='hidden'),
)


def _free_port() -> int:
    """Return a free TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        return int(probe.getsockname()[1])


def _viewer_executable() -> str:
    """Return the Rerun command installed next to the running interpreter, or the one on `PATH`."""
    found = shutil.which('rerun', path=str(Path(sys.executable).parent)) or shutil.which('rerun')
    if found is None:
        error_msg = 'The `rerun` command is not installed. Install `rerun-sdk` in the environment of the build.'
        raise GalleryError(error_msg)
    return found


def _start_viewer(arguments: list[str], home: Path) -> subprocess.Popen[bytes]:
    """Start the Rerun command in its own process group, with a fresh state directory and no info notifications."""
    environment = {**os.environ, 'RUST_LOG': 'warn', 'HOME': str(home), 'RERUN_ANALYTICS': 'false'}
    return subprocess.Popen(  # noqa: S603  # nosec B603: runs the installed Rerun viewer with fixed arguments
        [_viewer_executable(), *arguments],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _stop_viewer(process: subprocess.Popen[bytes]) -> None:
    """Stop the Rerun command and the viewer it started."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    process.wait()


def _first_view(part: rrb.View | rrb.Container) -> rrb.View:
    """Return the first view of a layout, its primary view."""
    if isinstance(part, rrb.View):
        return part
    for child in part.contents:
        return _first_view(child)
    error_msg = 'The blueprint has no view. Give the example a blueprint with at least one view.'
    raise GalleryError(error_msg)


def thumbnail_blueprint(blueprint: rrb.Blueprint) -> rrb.Blueprint:
    """Return a blueprint with the primary view of a blueprint alone, without its title bar and the panels.

    Args:
        blueprint:
            The blueprint of an example, whose first view is its primary view.

    Returns:
        blueprint:
            The blueprint of the thumbnail.

    Raises:
        GalleryError:
            The blueprint has no view.

    Examples:
        >>> import rerun.blueprint as rrb
        >>> from gallery_tools import thumbnail_blueprint
        >>> layout = rrb.Blueprint(rrb.Horizontal(rrb.Spatial2DView(origin='/u'), rrb.Spatial2DView(origin='/v')))
        >>> view = thumbnail_blueprint(layout).root_container.contents[0]
        >>> str(view.origin), view.titlebar
        ('/u', False)
    """
    view = copy.copy(_first_view(blueprint.root_container))
    view.titlebar = False
    return rrb.Blueprint(view, *HIDDEN_PANELS)


def _views_of(blueprint: rrb.Blueprint) -> int:
    """Return the number of views of a blueprint."""
    return len(view_boxes(blueprint.root_container, (1, 1)))


class _QuietHandler(SimpleHTTPRequestHandler):
    """File handler that allows requests from the web viewer's origin and logs nothing."""

    def end_headers(self) -> None:
        """Allow every origin, since the web viewer is served from another port."""
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ANN401
        """Log nothing."""


class Capturer:
    """Base class of capture backends, which turn a recording and a blueprint into an image of its views.

    Args:
        size:
            The logical size of the viewer window.

        settings:
            The capture settings.
    """

    name = 'base'

    def __init__(self, size: tuple[int, int], settings: CaptureSettings) -> None:
        self.size = size
        self.settings = settings

    def __enter__(self) -> Self:
        """Start the viewer."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the viewer."""

    def capture(self, recording: Path, blueprint: rrb.Blueprint, application_id: str, wait: float) -> Image.Image:
        """Return an image of the views of a recording at its last frame, laid out by a blueprint.

        Args:
            recording:
                The `.rrd` file.

            blueprint:
                The blueprint, without panels.

            application_id:
                The application id of the recording.

            wait:
                The seconds to wait for the recording to load and the views to render.

        Returns:
            image:
                The image of the viewer window.

        Raises:
            NotImplementedError:
                The subclass does not implement the method.
        """
        raise NotImplementedError


class NativeCapturer(Capturer):
    """Backend that drives the native Rerun viewer in headless mode through its viewer client.

    The viewer runs with a fresh state directory and logs only warnings, so it draws no info notifications. The capture
    waits until the viewer reports no source loading and its timelines and views stop changing, sends the blueprint to
    the loaded recording, moves the time cursor to the last frame, and saves a screenshot of the window.

    Args:
        size:
            The logical size of the viewer window.

        settings:
            The capture settings.
    """

    name = 'native'

    def __enter__(self) -> Self:
        """Start the headless viewer and wait until it answers."""
        self._home = Path(tempfile.mkdtemp(prefix='skcomplex-rerun-'))
        self._port = _free_port()
        width, height = self.size
        arguments = ['--headless', '--port', str(self._port), '--hide-welcome-screen', '--window-size']
        self._process = _start_viewer([*arguments, f'{width}x{height}'], self._home)
        self._url = f'rerun+http://127.0.0.1:{self._port}/proxy'
        self._client = ViewerClient.connect(self._url)
        deadline = time.monotonic() + STARTUP_SECONDS
        while True:
            try:
                self._client.viewer_state()
            except ConnectionError:
                if time.monotonic() > deadline or self._process.poll() is not None:
                    self._close()
                    error_msg = 'The headless Rerun viewer did not start. Check that a graphics adapter is available.'
                    raise GalleryError(error_msg) from None
                time.sleep(POLL_SECONDS)
            else:
                return self

    def _close(self) -> None:
        """Stop the viewer and remove its state directory."""
        _stop_viewer(self._process)
        shutil.rmtree(self._home, ignore_errors=True)

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the viewer."""
        self._close()

    def _wait_stable(self, wait: float, done: Callable[[ViewerState], bool]) -> ViewerState:
        """Return the viewer state once nothing loads, a condition holds, and the timelines and views stop changing."""
        deadline = time.monotonic() + wait
        previous: object = None
        stable = 0
        while True:
            state = self._client.viewer_state()
            key = (
                [(timeline.name, timeline.end) for item in state.recordings for timeline in item.timelines],
                sorted(view.view_id for view in state.views),
            )
            stable = stable + 1 if key == previous and not state.loading and done(state) else 0
            previous = key
            if stable >= STABLE_POLLS:
                return state
            if time.monotonic() > deadline:
                error_msg = f'The recording did not finish loading within {wait} seconds.'
                raise GalleryError(error_msg)
            time.sleep(POLL_SECONDS)

    def _send(self, blueprint: rrb.Blueprint, store_id: str) -> None:
        """Send a blueprint for the recording the viewer loaded, under the ids the viewer gave it."""
        _, application_id, recording_id = store_id.split(':', 2)
        stream = rr.RecordingStream(application_id.replace('\\:', ':'), recording_id=recording_id)
        stream.connect_grpc(self._url)
        stream.send_blueprint(blueprint, make_active=True, make_default=True)
        stream.flush()
        stream.disconnect()

    def _move_to_end(self, state: ViewerState, store_id: str) -> None:
        """Move the time cursor to the last frame of the timeline the viewer shows."""
        for item in state.recordings:
            for timeline in item.timelines:
                if timeline.name != (item.current_timeline or TIMELINE) or timeline.end is None:
                    continue
                if timeline.time_type == 'duration':
                    self._client.set_time(timeline.name, duration=timeline.end / NANOSECONDS, recording=store_id)
                else:
                    self._client.set_time(timeline.name, sequence=timeline.end, recording=store_id)

    def capture(self, recording: Path, blueprint: rrb.Blueprint, application_id: str, wait: float) -> Image.Image:
        """Return an image of the views of a recording at its last frame, laid out by a blueprint.

        Args:
            recording:
                The `.rrd` file.

            blueprint:
                The blueprint, without panels.

            application_id:
                The application id of the recording, which the viewer replaces with the id of its catalog.

            wait:
                The seconds to wait for the recording to load and the views to render.

        Returns:
            image:
                The image of the viewer window.
        """
        self._client.close_recordings('all')
        self._client.open_url(str(Path(recording).absolute()))
        state = self._wait_stable(wait, lambda state: state.active_recording is not None)
        store_id = str(state.active_recording)
        self._send(blueprint, store_id)
        expected = _views_of(blueprint)
        state = self._wait_stable(wait, lambda state: len(state.views) == expected)
        self._move_to_end(state, store_id)
        time.sleep(self.settings.settle)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'screenshot.png'
            self._client.save_screenshot(str(path))
            with Image.open(path) as image:
                return image.convert('RGB')


class WebCapturer(Capturer):
    """Backend that loads the Rerun web viewer of the installed version in headless Chromium through Playwright.

    The recording and the blueprint are served over HTTP to the web viewer, which the Rerun command serves. The capture
    waits for the page to load and for the canvas to stop changing, and takes a screenshot of the canvas. The page has a
    device pixel ratio of 1, since the web viewer doubles its interface on denser pages, so both backends draw the views
    at the same logical size.

    Args:
        size:
            The logical size of the browser viewport.

        settings:
            The capture settings.
    """

    name = 'web'

    def __enter__(self) -> Self:
        """Start the file server, the web viewer server, and the browser."""
        from playwright.sync_api import sync_playwright  # noqa: PLC0415  # optional: only the web backend needs it

        self._folder = Path(tempfile.mkdtemp(prefix='skcomplex-web-'))
        handler = functools.partial(_QuietHandler, directory=str(self._folder))
        self._server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self._viewer_port = _free_port()
        arguments = ['--serve-web', '--web-viewer-port', str(self._viewer_port), '--port', str(_free_port())]
        self._process = _start_viewer(arguments, self._folder)
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(args=['--enable-unsafe-swiftshader'])
        width, height = self.size
        self._page = self._browser.new_page(
            viewport={'width': width, 'height': height},
            device_scale_factor=1,
            color_scheme='dark',
        )
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the browser and the servers."""
        self._browser.close()
        self._playwright.stop()
        self._server.shutdown()
        _stop_viewer(self._process)
        shutil.rmtree(self._folder, ignore_errors=True)

    def _stable_image(self, wait: float) -> bytes:
        """Return a screenshot of the canvas once it has stayed the same for several polls."""
        canvas = self._page.locator('canvas').first
        canvas.wait_for(timeout=wait * 1000)
        deadline = time.monotonic() + wait
        previous, stable = b'', 0
        while True:
            image = canvas.screenshot()
            stable = stable + 1 if image == previous else 0
            previous = image
            if stable >= STABLE_POLLS - 1 or time.monotonic() > deadline:
                return image
            time.sleep(self.settings.settle)

    def capture(self, recording: Path, blueprint: rrb.Blueprint, application_id: str, wait: float) -> Image.Image:
        """Return an image of the views of a recording at its last frame, laid out by a blueprint.

        Args:
            recording:
                The `.rrd` file.

            blueprint:
                The blueprint, without panels.

            application_id:
                The application id of the recording, which the blueprint must match.

            wait:
                The seconds to wait for the recording to load and the views to render.

        Returns:
            image:
                The image of the canvas.
        """
        stem = f'{time.monotonic_ns()}'
        shutil.copyfile(recording, self._folder / f'{stem}.rrd')
        blueprint.save(application_id, self._folder / f'{stem}.rbl')
        base = f'http://127.0.0.1:{self._server.server_address[1]}'
        query = f'url={base}/{stem}.rrd&url={base}/{stem}.rbl&renderer=webgl'
        self._page.goto(f'http://127.0.0.1:{self._viewer_port}/?{query}')
        self._page.wait_for_load_state('networkidle', timeout=wait * 1000)
        with Image.open(io.BytesIO(self._stable_image(wait))) as image:
            return image.convert('RGB')


CAPTURERS: dict[str, type[Capturer]] = {'native': NativeCapturer, 'web': WebCapturer}


def _fitted(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Return an image without background bands, scaled to fill a size and cropped to its ratio around the centre."""
    return ImageOps.fit(trim_border(image), size, method=Image.Resampling.LANCZOS)


def _checked(
    capturer: Capturer,
    recording: Path,
    blueprint: rrb.Blueprint,
    application_id: str,
    size: tuple[int, int],
) -> Image.Image:
    """Return a capture that passes the checks, retrying once with a longer wait."""
    settings = capturer.settings
    boxes = view_boxes(blueprint.root_container, size)
    problems: list[str] = []
    for attempt in (1, 2):
        try:
            image = _fitted(capturer.capture(recording, blueprint, application_id, settings.wait * attempt), size)
        except GalleryError as error:
            problems = [str(error)]
            continue
        problems = find_problems(image, size, boxes)
        if not problems:
            return image
    error_msg = f'The capture of {recording.name} failed its checks twice: {"; ".join(problems)}.'
    raise GalleryError(error_msg)


def capture_thumbnail(
    recording: Path,
    blueprint: rrb.Blueprint,
    application_id: str,
    folder: Path,
    *,
    backend: str = 'native',
    settings: CaptureSettings | None = None,
) -> Path:
    """Capture the primary view of a recording at its last frame, check it, and write it as the PNG thumbnail.

    Args:
        recording:
            The `.rrd` file.

        blueprint:
            The blueprint of the example, whose first view is its primary view.

        application_id:
            The application id of the recording.

        folder:
            The folder to write `thumbnail.png` to.

        backend:
            The capture backend, `'native'` or `'web'`.

        settings:
            The capture settings, or `None` for the defaults.

    Returns:
        thumbnail:
            The written thumbnail.

    Raises:
        GalleryError:
            The backend is unknown, or a capture fails its checks twice.
    """
    if backend not in BACKENDS:
        error_msg = f'backend = {backend!r} is not one of {", ".join(BACKENDS)}.'
        raise GalleryError(error_msg)
    settings = settings or CaptureSettings()
    size = settings.thumbnail_size
    with CAPTURERS[backend](settings.logical(size), settings) as capturer:
        image = _checked(capturer, recording, thumbnail_blueprint(blueprint), application_id, size)
    return encode_png(image, folder / 'thumbnail.png', settings.max_png_bytes)
