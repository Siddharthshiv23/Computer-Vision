"""Turn a Report into HTML.

Images go inline as data URIs so results never pass through Gradio's file cache.
All text from files, pipelines or errors is escaped here.
"""
from __future__ import annotations

import base64
import io
from html import escape

import numpy as np
from PIL import Image

from app.adapter import Slot
from app.processing import Outcome, Report

STAGE_THUMB_WIDTH = 640
# key, heading, decimals, higher is better
METRICS = (
    ("psnr", "PSNR (dB)", 2, True),
    ("ssim", "SSIM", 3, True),
    ("delta_e2000", "DeltaE2000", 2, False),
)


def _data_uri(rgb: np.ndarray, max_width: int | None = None, quality: int = 92) -> str:
    image = Image.fromarray(rgb)
    if max_width and image.width > max_width:
        image = image.resize((max_width, max(1, image.height * max_width // image.width)), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _ms(value: float) -> str:
    return f"{value:,.0f} ms" if value >= 10 else f"{value:.1f} ms"


def _empty(kind: str, title: str, detail: str = "") -> str:
    text = f"<span>{escape(detail)}</span>" if detail else ""
    return f'<div class="isp-empty {kind}"><strong>{escape(title)}</strong>{text}</div>'


def _shot(outcome: Outcome) -> str:
    time_text = f'<span class="isp-time">{_ms(outcome.total_ms)}</span>' if outcome.status == "done" else ""
    caption = f"<figcaption><h3>{escape(outcome.label)}</h3>{time_text}</figcaption>"
    if outcome.status == "done":
        alt = escape(f"{outcome.label} result")
        body = (
            f'<img src="{_data_uri(outcome.rgb)}" alt="{alt}" data-isp="{escape(outcome.name)}" '
            'onload="window.ispPeek && ispPeek(this)" onclick="window.ispPeek && ispPeek(this, event)">'
        )
        if outcome.info.get("fallback"):
            body += '<p class="isp-flag">The network output was invalid, so this pipeline fell back to the classical result.</p>'
    elif outcome.status == "unavailable":
        body = _empty("", outcome.message)
    elif outcome.status == "failed":
        body = _empty("isp-failed", "No result", outcome.message)
    elif outcome.status == "running":
        body = _empty("isp-running", "Developing")
    else:
        body = _empty("", "Waiting")
    return f'<figure class="isp-shot">{caption}{body}</figure>'


def _grid(outcomes: list[Outcome]) -> str:
    return f'<div class="isp-grid">{"".join(_shot(o) for o in outcomes)}</div>'


def _crop(outcome: Outcome) -> str:
    if outcome.status != "done":
        return '<figure class="isp-crop"></figure>'  # holds the column so crops sit under their image
    name, label = escape(outcome.name), escape(outcome.label)
    return (
        f'<figure class="isp-crop"><div data-isp-crop="{name}" role="img" aria-label="{label} crop at full size"></div>'
        f"<figcaption>{label}</figcaption></figure>"
    )


def _detail(outcomes: list[Outcome]) -> str:
    return (
        "<h2>Detail at full size</h2>"
        '<p class="isp-soft">The same area from each result. Select a point on any image above to move it.</p>'
        f'<div class="isp-crops">{"".join(_crop(o) for o in outcomes)}</div>'
    )


def _quality(report: Report, done: list[Outcome]) -> str:
    head = "<h2>Quality against your reference</h2>"
    if not report.has_reference:
        return head + '<p class="isp-soft">Add a reference image to compare PSNR, SSIM and DeltaE2000. Without one, no scores are shown.</p>'
    scored = [o for o in done if o.metrics]
    if not scored:
        return head + f'<p class="isp-soft">{escape(report.metrics_note or "No scores could be computed for this image.")}</p>'
    best = {}
    for key, _, _, higher in METRICS:
        values = [o.metrics[key] for o in scored if key in o.metrics]
        if len(values) > 1:
            best[key] = max(values) if higher else min(values)
    columns = "".join(f'<th scope="col">{escape(title)}</th>' for _, title, _, _ in METRICS)
    rows = ""
    for outcome in scored:
        cells = ""
        for key, _, digits, _ in METRICS:
            value = outcome.metrics.get(key)
            text = "not computed" if value is None else f"{value:.{digits}f}"
            cells += f'<td{" class=isp-best" if value is not None and value == best.get(key) else ""}>{text}</td>'
        rows += f'<tr><th scope="row">{escape(outcome.label)}</th>{cells}</tr>'
    return (
        head
        + '<p class="isp-soft">Higher is better for PSNR and SSIM, lower for DeltaE2000. Best values are bold. Measured on this preview only.</p>'
        + f'<div class="isp-scroll"><table class="isp-table"><thead><tr><th scope="col">Pipeline</th>{columns}</tr></thead>'
        + f"<tbody>{rows}</tbody></table></div>"
    )


def _latency(done: list[Outcome]) -> str:
    columns = ""
    for outcome in done:
        # The heading already carries the measured total, so a pipeline's own "total" is not a stage.
        timings = {name: ms for name, ms in outcome.timings_ms.items() if name != "total"}
        peak = max(timings.values(), default=0) or 1
        items = "".join(
            f'<li><div class="isp-row"><span>{escape(str(name))}</span><span>{_ms(ms)}</span></div>'
            f'<div class="isp-bar" style="width:{ms / peak * 100:.1f}%"></div></li>'
            for name, ms in timings.items()
        )
        stages = f"<ol>{items}</ol>" if items else '<p class="isp-soft">This pipeline reported no stage timings.</p>'
        columns += f"<div><h3>{escape(outcome.label)}: {_ms(outcome.total_ms)}</h3>{stages}</div>"
    return f'<h2>Latency</h2><p class="isp-soft">Measured on this server for this image.</p><div class="isp-lat">{columns}</div>'


def _stages(done: list[Outcome]) -> str:
    blocks = ""
    for outcome in done:
        if not outcome.stages:
            continue
        thumbs = ""
        for name, image in outcome.stages.items():
            ms = outcome.timings_ms.get(name)
            time_text = f"<span>{_ms(ms)}</span>" if ms is not None else ""
            thumbs += (
                f'<figure><img loading="lazy" src="{_data_uri(image, STAGE_THUMB_WIDTH, 80)}" alt="{escape(str(name))} stage output">'
                f"<figcaption><span>{escape(str(name))}</span>{time_text}</figcaption></figure>"
            )
        blocks += f'<details><summary>{escape(outcome.label)}</summary><div class="isp-thumbs">{thumbs}</div></details>'
    return f"<h2>Stage by stage</h2>{blocks}" if blocks else ""


def _status(report: Report) -> str:
    parts = [report.status + "..." if report.status else ""]
    if report.filename:
        parts.insert(0, report.filename)
    if report.note:
        parts.append(report.note)
    text = " ".join(p if p.endswith((".", "...")) else p + "." for p in parts if p)
    return f'<p class="isp-status" role="status" aria-live="polite">{escape(text)}</p>'


def page(report: Report) -> str:
    alert = ""
    if report.error:
        alert = f'<div class="isp-alert" role="alert"><strong>That did not work</strong>{escape(report.error)}</div>'
    body = _status(report) + alert + _grid(report.outcomes)
    done = [o for o in report.outcomes if o.status == "done"]
    if done and not report.status:
        body += _detail(report.outcomes) + _quality(report, done) + _latency(done) + _stages(done)
    return f'<div class="isp-out">{body}</div>'


def idle(slots: list[Slot]) -> str:
    """The page before any upload: states what is ready, shows no image."""
    outcomes = [
        Outcome(s.name, s.label, "waiting" if s.pipeline else "unavailable", s.reason) for s in slots
    ]
    return page(Report(outcomes=outcomes))
