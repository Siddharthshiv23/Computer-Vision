"""Input handling and honest-state behaviour of the app.

Stub pipelines here exist only to drive the app's code paths; they are never
reachable from the app itself.
"""
from __future__ import annotations

import io
import sys
import types

import numpy as np
import pytest
from PIL import Image

from app import adapter, processing, render
from app.adapter import Slot
from isp.types import InputError, PipelineResult, RawImage


def raw_of(h, w, seed=0):
    return RawImage(bayer=np.random.default_rng(seed).random((h, w), dtype=np.float32))


def encode(mode, size=(64, 48), fmt="PNG", color=None):
    buf = io.BytesIO()
    Image.new(mode, size, color if color is not None else 128).save(buf, fmt)
    return buf.getvalue()


class Echo:
    """Stub: grey image the size of the mosaic."""

    name = "stub"

    def __init__(self, value=0.5, info=None):
        self.value, self.info = value, info or {}

    def run(self, raw):
        h, w = raw.bayer.shape
        rgb = np.full((h, w, 3), self.value, np.float32)
        return PipelineResult(
            rgb=rgb,
            stages={"demosaic": rgb, "mask": rgb[..., 0]},
            timings_ms={"demosaic": 2.0, "tone": 6.0},
            info=self.info,
        )


class Raises:
    name = "stub"

    def __init__(self, exc):
        self.exc = exc

    def run(self, raw):
        raise self.exc


def slot(pipeline, name="hybrid"):
    return Slot(name=name, label=name.title(), pipeline=pipeline, reason="")


def missing(name="learned"):
    return Slot(name=name, label=name.title(), pipeline=None, reason="Not available yet.")


def upload(tmp_path, data=b"bytes", name="photo.jpg"):
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def load_ok(data, filename=None):
    return raw_of(64, 64)


def steps(path, slots, load=load_ok, load_reason="", metrics=None, metrics_reason="no metrics", ref=None):
    return list(
        processing.develop(
            path, ref, slots=slots, load=load, load_reason=load_reason, metrics=metrics, metrics_reason=metrics_reason
        )
    )


def final(*args, **kwargs):
    return steps(*args, **kwargs)[-1]


# --- adapter ---------------------------------------------------------------


def test_resolve_reports_unlanded_module_as_not_available():
    obj, reason = adapter.resolve("isp.not_landed_module", "Thing")
    assert obj is None
    assert "not available yet" in reason.lower()


def test_resolve_names_a_missing_third_party_dependency(monkeypatch):
    def fake_import(name):
        raise ModuleNotFoundError("No module named 'rawpy'", name="rawpy")

    monkeypatch.setattr(adapter.importlib, "import_module", fake_import)
    obj, reason = adapter.resolve("isp.io", "load_input")
    assert obj is None
    assert "rawpy" in reason


def test_resolve_reports_missing_attribute_as_not_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "isp_test_empty", types.ModuleType("isp_test_empty"))
    obj, reason = adapter.resolve("isp_test_empty", "Thing")
    assert obj is None
    assert "not available yet" in reason.lower()


def test_load_slots_returns_all_three_in_order_when_weights_are_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ISP_LEARNED_WEIGHTS", str(tmp_path / "absent.onnx"))
    monkeypatch.setenv("ISP_HYBRID_WEIGHTS", str(tmp_path / "absent.onnx"))
    monkeypatch.setattr(adapter, "resolve", lambda module, attr: (lambda *a: object(), ""))
    slots = adapter.load_slots()
    assert [s.name for s in slots] == ["classical", "learned", "hybrid"]
    assert slots[0].pipeline is not None
    assert slots[1].pipeline is None and "weights" in slots[1].reason.lower()
    assert slots[2].pipeline is None and "weights" in slots[2].reason.lower()


def test_load_slots_turns_constructor_failure_into_a_reason(monkeypatch, tmp_path):
    def boom(*args):
        raise RuntimeError("corrupt weights")

    weights = tmp_path / "w.onnx"
    weights.write_bytes(b"x")
    monkeypatch.setenv("ISP_HYBRID_WEIGHTS", str(weights))
    monkeypatch.setattr(adapter, "resolve", lambda module, attr: (boom, ""))
    hybrid = adapter.load_slots()[2]
    assert hybrid.pipeline is None
    assert "could not start" in hybrid.reason.lower()


def test_load_metrics_needs_all_three_functions(monkeypatch):
    def only_psnr(module, attr):
        return (len, "") if attr == "psnr" else (None, "Not available yet.")

    monkeypatch.setattr(adapter, "resolve", only_psnr)
    fns, reason = adapter.load_metrics()
    assert fns is None and reason


# --- preview downscale -----------------------------------------------------


def test_small_mosaic_is_left_alone():
    raw = raw_of(64, 96)
    out, note = processing.prepare_raw(raw)
    assert out.bayer is raw.bayer
    assert note == ""


def test_large_mosaic_is_reduced_to_at_most_2mp_with_even_dimensions():
    out, note = processing.prepare_raw(raw_of(3000, 4000))
    h, w = out.bayer.shape
    assert h * w <= processing.MAX_PREVIEW_PIXELS
    assert h * w > 0.95 * processing.MAX_PREVIEW_PIXELS
    assert h % 2 == 0 and w % 2 == 0
    assert out.bayer.dtype == np.float32
    assert "4000" in note and str(w) in note


def test_downscale_keeps_the_rggb_phase():
    bayer = np.zeros((2400, 2400), np.float32)
    bayer[0::2, 0::2] = 0.9  # R
    bayer[0::2, 1::2] = 0.5  # G
    bayer[1::2, 0::2] = 0.4  # G
    bayer[1::2, 1::2] = 0.1  # B
    small = processing.prepare_raw(RawImage(bayer=bayer))[0].bayer
    np.testing.assert_allclose(small[0::2, 0::2], 0.9, atol=1e-5)
    np.testing.assert_allclose(small[0::2, 1::2], 0.5, atol=1e-5)
    np.testing.assert_allclose(small[1::2, 0::2], 0.4, atol=1e-5)
    np.testing.assert_allclose(small[1::2, 1::2], 0.1, atol=1e-5)


def test_downscale_keeps_camera_metadata():
    gains, ccm = np.array([2.0, 1.0, 1.5]), np.eye(3)
    raw = RawImage(bayer=np.zeros((2400, 2400), np.float32), wb_gains=gains, ccm=ccm, meta={"iso": 100})
    out = processing.prepare_raw(raw)[0]
    assert out.wb_gains is gains and out.ccm is ccm
    assert out.meta["iso"] == 100


def test_odd_mosaic_is_cropped_to_even():
    assert processing.prepare_raw(raw_of(65, 97))[0].bayer.shape == (64, 96)


def test_tiny_mosaic_is_rejected():
    with pytest.raises(InputError, match="32"):
        processing.prepare_raw(raw_of(30, 64))


def test_extreme_aspect_that_would_shrink_below_minimum_is_rejected():
    with pytest.raises(InputError, match="32"):
        processing.prepare_raw(raw_of(40, 100000))


def test_non_finite_mosaic_is_rejected():
    raw = raw_of(64, 64)
    raw.bayer[3, 3] = np.nan
    with pytest.raises(InputError):
        processing.prepare_raw(raw)


# --- uploads are not kept --------------------------------------------------


def test_upload_is_deleted_after_reading(tmp_path):
    path = upload(tmp_path, b"abc")
    assert processing.read_and_delete(path) == b"abc"
    assert list(tmp_path.iterdir()) == []


def test_vanished_upload_asks_for_a_new_upload(tmp_path):
    with pytest.raises(InputError, match="again"):
        processing.read_and_delete(str(tmp_path / "gone.jpg"))


def test_both_uploads_are_deleted_even_when_the_image_is_rejected(tmp_path):
    def reject(data, filename=None):
        raise InputError("nope")

    report = final(upload(tmp_path), [slot(Echo())], load=reject, ref=upload(tmp_path, encode("RGB"), "ref.png"))
    assert report.error == "nope"
    assert list(tmp_path.iterdir()) == []


def test_uploads_are_deleted_while_the_loader_is_unavailable(tmp_path):
    report = final(upload(tmp_path), [missing()], load=None, load_reason="Image loading is not available yet.")
    assert "not available yet" in report.error
    assert list(tmp_path.iterdir()) == []


# --- reference images ------------------------------------------------------


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P", "LA"])
def test_reference_modes_become_float_rgb(mode):
    color = 128 if mode in ("L", "P") else (128,) * len(mode)
    ref = processing.read_reference(encode(mode, color=color))
    assert ref.shape == (48, 64, 3)
    assert ref.dtype == np.float32
    assert 0.0 <= ref.min() and ref.max() <= 1.0


def test_sixteen_bit_reference_keeps_its_range():
    buf = io.BytesIO()
    Image.fromarray(np.full((48, 64), 32768, np.uint16)).save(buf, "PNG")
    ref = processing.read_reference(buf.getvalue())
    assert ref.shape == (48, 64, 3)
    np.testing.assert_allclose(ref, 0.5, atol=1e-3)


def test_cmyk_reference_is_converted():
    ref = processing.read_reference(encode("CMYK", fmt="JPEG", color=(0, 0, 0, 0)))
    assert ref.shape == (48, 64, 3)


def test_corrupt_reference_is_rejected_kindly():
    with pytest.raises(InputError, match="reference"):
        processing.read_reference(b"this is not an image")


def test_truncated_reference_is_rejected_kindly():
    noise = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noise).save(buf, "JPEG")
    with pytest.raises(InputError, match="reference"):
        processing.read_reference(buf.getvalue()[:2000])


def test_oversized_reference_is_rejected_before_decoding(monkeypatch):
    monkeypatch.setattr(processing, "MAX_REFERENCE_PIXELS", 100)
    with pytest.raises(InputError, match="reference"):
        processing.read_reference(encode("RGB"))


def test_reference_follows_preview_downscale():
    ref = np.full((3000, 4000, 3), 0.25, np.float32)
    fitted = processing.fit_reference(ref, (1224, 1632))
    assert fitted.shape == (1224, 1632, 3)
    np.testing.assert_allclose(fitted, 0.25, atol=1e-5)


def test_odd_reference_is_cropped_not_resampled():
    ref = np.random.default_rng(0).random((65, 97, 3)).astype(np.float32)
    np.testing.assert_array_equal(processing.fit_reference(ref, (64, 96)), ref[:64, :96])


def test_reference_of_another_shape_is_rejected():
    with pytest.raises(InputError, match="reference"):
        processing.fit_reference(np.zeros((100, 100, 3), np.float32), (64, 128))


# --- displayable images ----------------------------------------------------


def test_to_display_handles_float_gray_and_uint8():
    assert processing.to_display(np.full((4, 4), 1.0, np.float32)).shape == (4, 4, 3)
    assert processing.to_display(np.full((4, 4, 1), 0.5, np.float32)).shape == (4, 4, 3)
    assert processing.to_display(np.full((4, 4, 3), 200, np.uint8)).max() == 200
    assert processing.to_display(np.full((4, 4, 3), 2.0, np.float32)).max() == 255


@pytest.mark.parametrize(
    "bad",
    [np.full((4, 4, 3), np.nan, np.float32), np.zeros((4, 4, 5), np.float32), np.zeros((4,), np.float32)],
)
def test_to_display_refuses_what_it_cannot_show_truthfully(bad):
    assert processing.to_display(bad) is None


# --- pipeline runs ---------------------------------------------------------


def test_no_upload_asks_for_one():
    report = final(None, [slot(Echo())])
    assert "choose" in report.error.lower()


def test_unavailable_pipeline_produces_no_image_while_others_run(tmp_path):
    gone, ok = final(upload(tmp_path), [missing(), slot(Echo())]).outcomes
    assert gone.status == "unavailable" and gone.rgb is None
    assert gone.message == "Not available yet."
    assert ok.status == "done"


def test_each_pipeline_reports_as_it_finishes(tmp_path):
    states = [
        [o.status for o in report.outcomes]  # the report is mutated in place, so snapshot it per step
        for report in processing.develop(
            upload(tmp_path),
            None,
            slots=[slot(Echo(), "classical"), slot(Echo())],
            load=load_ok,
            load_reason="",
            metrics=None,
            metrics_reason="",
        )
    ]
    assert ["done", "running"] in states
    assert states[-1] == ["done", "done"]


def test_filename_reaches_the_loader(tmp_path):
    seen = {}

    def load(data, filename=None):
        seen.update(data=data, filename=filename)
        return raw_of(64, 64)

    final(upload(tmp_path, b"xyz", "shot.dng"), [slot(Echo())], load=load)
    assert seen == {"data": b"xyz", "filename": "shot.dng"}


def test_loader_crash_is_a_friendly_error_not_a_traceback(tmp_path):
    def load(data, filename=None):
        raise RuntimeError("secret internal path C:/x")

    report = final(upload(tmp_path), [slot(Echo())], load=load)
    assert report.error
    assert "secret" not in report.error


def test_one_failing_pipeline_leaves_the_others_shown(tmp_path):
    bad, good = final(upload(tmp_path), [slot(Raises(RuntimeError("boom x")), "learned"), slot(Echo())]).outcomes
    assert bad.status == "failed" and bad.rgb is None
    assert "boom" not in bad.message
    assert good.status == "done" and good.rgb is not None


def test_input_error_from_a_pipeline_is_shown_verbatim(tmp_path):
    outcome = final(upload(tmp_path), [slot(Raises(InputError("X-Trans is not supported.")))]).outcomes[0]
    assert outcome.status == "failed"
    assert outcome.message == "X-Trans is not supported."


def test_nan_output_is_not_displayed(tmp_path):
    outcome = final(upload(tmp_path), [slot(Echo(value=np.nan))]).outcomes[0]
    assert outcome.status == "failed" and outcome.rgb is None


def test_malformed_result_counts_as_a_failed_pipeline(tmp_path):
    class Malformed:
        def run(self, raw):
            return PipelineResult(rgb=None)

    outcome = final(upload(tmp_path), [slot(Malformed())]).outcomes[0]
    assert outcome.status == "failed" and outcome.rgb is None


def test_timings_come_from_the_pipeline_plus_a_measured_total(tmp_path):
    outcome = final(upload(tmp_path), [slot(Echo())]).outcomes[0]
    assert outcome.timings_ms == {"demosaic": 2.0, "tone": 6.0}
    assert outcome.total_ms > 0


# --- metrics ---------------------------------------------------------------

METRICS = {
    "psnr": lambda p, t: 31.0,
    "ssim": lambda p, t: 0.9,
    "delta_e2000": lambda p, t: 4.0,
}


def test_no_reference_means_no_metrics(tmp_path):
    assert final(upload(tmp_path), [slot(Echo())], metrics=METRICS).outcomes[0].metrics == {}


def test_metrics_are_computed_against_the_fitted_reference(tmp_path):
    seen = []

    def psnr(pred, target):
        seen.append((pred.shape, target.shape))
        return 31.0

    ref = upload(tmp_path, encode("RGB", (128, 128)), "ref.png")
    report = final(upload(tmp_path), [slot(Echo())], metrics={**METRICS, "psnr": psnr}, ref=ref)
    assert report.outcomes[0].metrics == {"psnr": 31.0, "ssim": 0.9, "delta_e2000": 4.0}
    assert seen == [((64, 64, 3), (64, 64, 3))]


def test_missing_metrics_module_gives_a_reason_and_no_numbers(tmp_path):
    ref = upload(tmp_path, encode("RGB", (64, 64)), "ref.png")
    report = final(upload(tmp_path), [slot(Echo())], metrics=None, ref=ref)
    assert report.outcomes[0].metrics == {}
    assert report.metrics_note == "no metrics"


def test_bad_reference_is_reported_before_any_pipeline_runs(tmp_path):
    ran = []

    class Tracks(Echo):
        def run(self, raw):
            ran.append(1)
            return super().run(raw)

    report = final(upload(tmp_path), [slot(Tracks())], metrics=METRICS, ref=upload(tmp_path, b"junk", "ref.png"))
    assert "reference" in report.error
    assert ran == []


def test_a_failing_metric_is_left_blank(tmp_path):
    def broken(pred, target):
        raise ValueError("x")

    ref = upload(tmp_path, encode("RGB", (64, 64)), "ref.png")
    report = final(upload(tmp_path), [slot(Echo())], metrics={**METRICS, "ssim": broken}, ref=ref)
    assert report.outcomes[0].metrics == {"psnr": 31.0, "delta_e2000": 4.0}


def test_nan_metric_is_left_blank(tmp_path):
    ref = upload(tmp_path, encode("RGB", (64, 64)), "ref.png")
    report = final(upload(tmp_path), [slot(Echo())], metrics={**METRICS, "ssim": lambda p, t: float("nan")}, ref=ref)
    assert "ssim" not in report.outcomes[0].metrics


# --- rendering -------------------------------------------------------------


def test_idle_page_says_which_pipelines_are_missing_and_shows_no_image():
    page = render.idle([missing("classical"), missing("learned"), missing("hybrid")])
    assert page.count("Not available yet") == 3
    assert "<img" not in page


def test_filename_is_escaped(tmp_path):
    report = final(upload(tmp_path), [slot(Echo())])
    report.filename = "<script>alert(1)</script>.jpg"
    assert "<script>alert" not in render.page(report)


def test_error_text_is_escaped():
    assert "<b>bad" not in render.page(processing.Report(error="<b>bad</b>"))


def test_results_page_shows_images_stages_latency_and_preview_note(tmp_path):
    report = final(upload(tmp_path), [slot(Echo())])
    report.note = "Previewing at 1632 x 1224."
    page = render.page(report)
    assert page.count("data:image/jpeg;base64,") == 3  # result + two stages
    assert "demosaic" in page and "tone" in page and "1632" in page


def test_metrics_table_shows_only_measured_values(tmp_path):
    ref = upload(tmp_path, encode("RGB", (64, 64)), "ref.png")
    page = render.page(final(upload(tmp_path), [slot(Echo())], metrics=METRICS, ref=ref))
    assert "31.00" in page and "0.900" in page and "4.00" in page


def test_no_metric_values_without_a_reference(tmp_path):
    page = render.page(final(upload(tmp_path), [slot(Echo())], metrics=METRICS))
    assert "<table" not in page
    assert "reference" in page.lower()


def test_crops_stay_under_their_own_column_when_a_pipeline_failed(tmp_path):
    report = final(upload(tmp_path), [slot(Echo(), "classical"), slot(Raises(RuntimeError()), "learned"), slot(Echo())])
    crops = render.page(report).split('class="isp-crops"')[1]
    assert crops.count('class="isp-crop"') == 3
    assert crops.index("classical") < crops.index("hybrid")
    assert "data-isp-crop=\"learned\"" not in crops


def test_fallback_is_disclosed(tmp_path):
    page = render.page(final(upload(tmp_path), [slot(Echo(info={"fallback": True}))]))
    assert "fell back" in page.lower()


# --- the Gradio shell ------------------------------------------------------


def test_handler_empties_the_pickers_and_restores_the_button(monkeypatch, tmp_path):
    from app import app as app_module

    monkeypatch.setattr(app_module, "_slots", lambda: [slot(Echo())])
    monkeypatch.setattr(adapter, "load_loader", lambda: (load_ok, ""))
    monkeypatch.setattr(adapter, "load_metrics", lambda: (None, "no metrics"))
    updates = list(app_module._develop(upload(tmp_path), None))
    assert updates[0][1]["interactive"] is False
    html, button, picked, reference = updates[-1]
    assert "<img" in html
    assert button["interactive"] is True
    assert picked is None and reference is None


def test_app_builds_without_any_pipeline():
    import gradio as gr

    from app.app import build_demo

    assert isinstance(build_demo(), gr.Blocks)
