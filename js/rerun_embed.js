// Open a gallery example's recording in an embedded Rerun web viewer, loaded only when the reader asks for it.

const RERUN_WEB_VIEWER = 'https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@0.39.0/+esm';

async function openRecording(link) {
  const container = document.createElement('div');
  container.className = 'skcomplex-rerun-viewer';
  link.replaceWith(container);
  const { WebViewer } = await import(RERUN_WEB_VIEWER);
  const recording = new URL(link.getAttribute('href'), window.location.href).href;
  const blueprint = recording.replace(/\.rrd$/, '.rbl');
  await new WebViewer().start([recording, blueprint], container, {
    width: '100%',
    height: '560px',
    hide_welcome_screen: true,
    theme: 'dark',
  });
}

document.addEventListener('click', (event) => {
  const link = event.target.closest('a.skcomplex-rerun');
  if (link === null) {
    return;
  }
  event.preventDefault();
  openRecording(link);
});
