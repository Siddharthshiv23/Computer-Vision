"""Gradio app: upload a RAW or JPEG and compare Classical, Learned and Hybrid.

Run with: python -m app.app
"""
from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from app import adapter, processing, render

MAX_UPLOAD = "100mb"
CSS = (Path(__file__).parent / "theme.css").read_text(encoding="utf-8")

# Moves the shared full-size crop. Runs when an image loads and when one is selected.
HEAD = """
<script>
window.ispPeek = function (img, event) {
  var box = img.getBoundingClientRect();
  var fx = event ? (event.clientX - box.left) / box.width : 0.5;
  var fy = event ? (event.clientY - box.top) / box.height : 0.5;
  document.querySelectorAll('img[data-isp]').forEach(function (source) {
    var crop = document.querySelector('[data-isp-crop="' + source.dataset.isp + '"]');
    if (!crop) return;
    var side = crop.clientWidth;
    var x = Math.max(0, Math.min(source.naturalWidth - side, fx * source.naturalWidth - side / 2));
    var y = Math.max(0, Math.min(source.naturalHeight - side, fy * source.naturalHeight - side / 2));
    crop.style.backgroundImage = 'url(' + source.src + ')';
    crop.style.backgroundSize = source.naturalWidth + 'px ' + source.naturalHeight + 'px';
    crop.style.backgroundPosition = -x + 'px ' + -y + 'px';
  });
};
</script>
"""

HEADER = """
<div class="isp-band" aria-hidden="true"></div>
<div class="isp-head">
  <h1>Hybrid ISP</h1>
  <p>Upload a RAW or JPEG photo and see how a classical pipeline, a neural network and a hybrid of the two develop it.</p>
</div>
"""

PRIVACY = (
    '<p class="isp-privacy">Your files are processed in memory. Each upload is deleted from the server '
    "as soon as it has been read, and results are not saved.</p>"
)


def _theme() -> gr.themes.Base:
    grey = {
        "body_background_fill": "#2b2b2b",
        "body_text_color": "#f1f1f1",
        "body_text_color_subdued": "#c4c4c4",
        "block_background_fill": "#363636",
        "block_border_color": "#565656",
        "block_label_background_fill": "#363636",
        "block_label_text_color": "#f1f1f1",
        "block_title_text_color": "#f1f1f1",
        "border_color_primary": "#565656",
        "input_background_fill": "#363636",
        "background_fill_primary": "#363636",
        "background_fill_secondary": "#363636",
        "table_even_background_fill": "#363636",
        "table_odd_background_fill": "#363636",
        "link_text_color": "#f1f1f1",
        "link_text_color_hover": "#ffffff",
        "button_primary_background_fill": "#f1f1f1",
        "button_primary_background_fill_hover": "#ffffff",
        "button_primary_text_color": "#1c1c1c",
        "button_primary_text_color_hover": "#1c1c1c",
    }
    both = {**grey, **{f"{key}_dark": value for key, value in grey.items()}}
    return gr.themes.Base(
        font=[gr.themes.GoogleFont("Archivo"), "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_none,
    ).set(**both)


def _develop(upload: str | None, reference: str | None):
    """Stream the report into the page, then empty the pickers, whose files are now gone."""
    load, load_reason = adapter.load_loader()
    metrics, metrics_reason = adapter.load_metrics()
    busy = gr.update(interactive=False, value="Developing")
    html = ""
    for report in processing.develop(
        upload,
        reference,
        slots=_slots(),
        load=load,
        load_reason=load_reason,
        metrics=metrics,
        metrics_reason=metrics_reason,
    ):
        html = render.page(report)
        yield html, busy, gr.update(), gr.update()
    yield html, gr.update(interactive=True, value="Develop image"), None, None


_cached_slots: list[adapter.Slot] = []


def _slots() -> list[adapter.Slot]:
    """Models load once per process."""
    if not _cached_slots:
        _cached_slots.extend(adapter.load_slots())
    return _cached_slots


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Hybrid ISP",
        fill_width=True,  # theme.css sets the real width, so idle and results pages match
        analytics_enabled=False,
        delete_cache=(600, 600),  # backstop for uploads that are never developed
    ) as demo:
        gr.HTML(HEADER)
        with gr.Row(equal_height=True):
            upload = gr.File(label="Photo: RAW (DNG) or JPEG", file_count="single", type="filepath", height=170)
            reference = gr.File(
                label="Reference image, optional, enables scores", file_count="single", type="filepath", height=170
            )
        button = gr.Button("Develop image", variant="primary", size="lg")
        gr.HTML(PRIVACY)
        results = gr.HTML(render.idle(_slots()))
        button.click(
            _develop,
            [upload, reference],
            [results, button, upload, reference],
            show_progress="hidden",
            concurrency_limit=1,
        )
    return demo


def launch_options() -> dict:
    return {"theme": _theme(), "css": CSS, "head": HEAD, "max_file_size": MAX_UPLOAD, "footer_links": []}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_demo().queue(max_size=16).launch(**launch_options())
