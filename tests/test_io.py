"""load_input across formats and the PRD risk list. DNGs are built byte by byte and decoded by real rawpy."""
import io
import struct
import time
import zlib

import numpy as np
import pytest
import rawpy
from PIL import Image

from isp.io import _camera_ccm, load_input
from isp.types import InputError, RawImage

# Hand-built mosaic: each CFA colour has its own level, so a wrong shift is visible at (0, 0).
R, G, B = 1000, 2000, 3000
SITE = {"R": R, "G": G, "B": B}


def mosaic(pattern, h=64, w=64):
    """uint16 mosaic where `pattern` ('BGGR' etc.) gives the colours of the top-left 2x2."""
    out = np.zeros((h, w), np.uint16)
    out[0::2, 0::2] = SITE[pattern[0]]
    out[0::2, 1::2] = SITE[pattern[1]]
    out[1::2, 0::2] = SITE[pattern[2]]
    out[1::2, 1::2] = SITE[pattern[3]]
    return out


def dng(pixels, cfa=(0, 1, 1, 2), cfa_dim=(2, 2), black=0, white=4095, orientation=1, neutral=(0.5, 1.0, 0.8)):
    """Minimal uncompressed little-endian DNG."""
    h, w = pixels.shape
    data = pixels.astype("<u2").tobytes()
    make, model = b"Test\0", b"Cam\0"
    matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    srational = b"".join(struct.pack("<ii", int(v * 10000), 10000) for v in matrix)
    rational = b"".join(struct.pack("<II", int(v * 10000), 10000) for v in neutral)
    entries = [
        (254, 4, 1, 0),
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, 1, 16),
        (259, 3, 1, 1),
        (262, 3, 1, 32803),
        (271, 2, len(make), make),
        (272, 2, len(model), model),
        (273, 4, 1, "DATA"),
        (274, 3, 1, orientation),
        (277, 3, 1, 1),
        (278, 4, 1, h),
        (279, 4, 1, len(data)),
        (284, 3, 1, 1),
        (33421, 3, 2, struct.pack("<HH", *cfa_dim)),
        (33422, 1, len(cfa), bytes(cfa)),
        (50706, 1, 4, bytes((1, 4, 0, 0))),
        (50708, 2, len(model), model),
        (50714, 4, 1, black),
        (50717, 4, 1, white),
        (50721, 10, 9, srational),
        (50728, 5, 3, rational),
        (50778, 3, 1, 21),
    ]
    table_size = 2 + 12 * len(entries) + 4
    extra_at = 8 + table_size
    extra = b""
    packed = []
    for tag, kind, count, value in entries:
        if isinstance(value, bytes):
            if len(value) <= 4:
                field = value.ljust(4, b"\0")
            else:
                field = struct.pack("<I", extra_at + len(extra))
                extra += value + b"\0" * (len(value) % 2)
        elif value == "DATA":
            field = None
        elif kind == 3:
            field = struct.pack("<HH", value, 0)
        else:
            field = struct.pack("<I", value)
        packed.append([tag, kind, count, field])
    data_at = extra_at + len(extra)
    body = struct.pack("<H", len(entries))
    for tag, kind, count, field in packed:
        body += struct.pack("<HHI", tag, kind, count) + (field or struct.pack("<I", data_at))
    return b"II*\0" + struct.pack("<I", 8) + body + struct.pack("<I", 0) + extra + data


def encode(image, fmt, **kw):
    buf = io.BytesIO()
    image.save(buf, fmt, **kw)
    return buf.getvalue()


def png16(array):
    return encode(Image.fromarray(array.astype(np.uint16)), "PNG")


def npy(array):
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def noise_rgb(h=64, w=64):
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def assert_contract(raw):
    assert isinstance(raw, RawImage)
    assert raw.bayer.dtype == np.float32 and raw.bayer.ndim == 2
    assert raw.bayer.shape[0] % 2 == 0 and raw.bayer.shape[1] % 2 == 0
    assert np.isfinite(raw.bayer).all()
    assert raw.bayer.min() >= 0 and raw.bayer.max() <= 1


def assert_rggb_levels(raw, scale):
    assert raw.bayer[0, 0] == pytest.approx(R / scale, abs=1e-4)
    assert raw.bayer[0, 1] == pytest.approx(G / scale, abs=1e-4)
    assert raw.bayer[1, 0] == pytest.approx(G / scale, abs=1e-4)
    assert raw.bayer[1, 1] == pytest.approx(B / scale, abs=1e-4)


# ---- DNG / RAW through real rawpy ----


def test_dng_from_bytes():
    raw = load_input(dng(mosaic("RGGB")), "shot.dng")
    assert_contract(raw)
    assert raw.bayer.shape == (64, 64)
    assert_rggb_levels(raw, 4095)


def test_dng_from_path(tmp_path):
    path = tmp_path / "shot.dng"
    path.write_bytes(dng(mosaic("RGGB")))
    assert_rggb_levels(load_input(str(path)), 4095)


def test_dng_without_a_filename_is_detected():
    assert_rggb_levels(load_input(dng(mosaic("RGGB"))), 4095)


@pytest.mark.parametrize(
    "pattern, cfa, shape",
    [
        ("BGGR", (2, 1, 1, 0), (62, 62)),
        ("GRBG", (1, 0, 2, 1), (64, 62)),
        ("GBRG", (1, 2, 0, 1), (62, 64)),
    ],
)
def test_dng_cfa_is_normalised_to_rggb(pattern, cfa, shape):
    raw = load_input(dng(mosaic(pattern), cfa=cfa), "shot.dng")
    assert_contract(raw)
    assert_rggb_levels(raw, 4095)
    assert raw.bayer.shape == shape


def test_dng_black_and_white_levels_are_removed():
    pixels = np.full((64, 64), 1064, np.uint16)
    raw = load_input(dng(pixels, black=64, white=2064), "shot.dng")
    assert raw.bayer[0, 0] == pytest.approx(0.5, abs=1e-4)


def test_dng_pixels_below_black_clip_to_zero():
    raw = load_input(dng(np.full((64, 64), 10, np.uint16), black=64), "shot.dng")
    assert raw.bayer.max() == 0.0


def test_dng_as_shot_white_balance_is_read():
    raw = load_input(dng(mosaic("RGGB"), neutral=(0.5, 1.0, 0.8)), "shot.dng")
    np.testing.assert_allclose(raw.wb_gains, [2.0, 1.0, 1.25], rtol=1e-3)


def test_dng_ccm_is_camera_to_srgb_with_white_preserved():
    raw = load_input(dng(mosaic("RGGB")), "shot.dng")
    assert raw.ccm.shape == (3, 3)
    np.testing.assert_allclose(raw.ccm @ np.ones(3), np.ones(3), atol=1e-4)


NIKON_XYZ2CAM = np.array([[0.8828, -0.2406, -0.0694], [-0.4874, 1.2603, 0.2541], [-0.066, 0.1509, 0.7587]])


def test_ccm_is_built_from_the_xyz_matrix_when_color_matrix_is_empty():
    # What LibRaw reports for real Nikon and Canon files: rgb_xyz_matrix filled, color_matrix all zero.
    ccm = _camera_ccm(NIKON_XYZ2CAM, np.zeros((3, 3)))
    np.testing.assert_allclose(ccm @ np.ones(3), np.ones(3), atol=1e-4)
    assert ccm[0, 0] > 1.2 and ccm[0, 1] < 0  # cameras are less saturated than sRGB


def test_ccm_falls_back_to_color_matrix():
    matrix = np.array([[1.5, -0.3, -0.2], [-0.2, 1.4, -0.2], [0.0, -0.4, 1.4]])
    np.testing.assert_allclose(_camera_ccm(np.zeros((3, 3)), matrix), matrix, rtol=1e-6)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.eye(3), np.full((3, 3), np.nan)], ids=["zero", "eye", "nan"])
def test_ccm_is_none_without_colour_data(matrix):
    assert _camera_ccm(np.zeros((3, 3)), matrix) is None


def test_dng_orientation_is_applied_keeping_rggb():
    pixels = mosaic("RGGB", 64, 96)
    raw = load_input(dng(pixels, orientation=6), "shot.dng")
    assert_contract(raw)
    assert raw.bayer.shape[0] > raw.bayer.shape[1]
    assert_rggb_levels(raw, 4095)


def test_x_trans_is_refused_by_name():
    cfa = tuple(np.random.default_rng(0).integers(0, 3, 36).tolist())
    with pytest.raises(InputError, match="X-Trans"):
        load_input(dng(mosaic("RGGB", 66, 66), cfa=cfa, cfa_dim=(6, 6)), "fuji.dng")


def test_corrupt_dng_is_refused():
    with pytest.raises(InputError):
        load_input(dng(mosaic("RGGB"))[:200], "shot.dng")


def test_damaged_raw_is_refused_rather_than_replaced_by_its_preview(monkeypatch):
    # CR2 and NEF are TIFF containers: Pillow can open the embedded preview if LibRaw gives up.
    # A real damaged CR2 cannot be committed, so LibRaw's verdict on one is injected here.
    def damaged(_):
        raise rawpy.LibRawDataError("data corrupted")

    monkeypatch.setattr(rawpy, "imread", damaged)
    with pytest.raises(InputError, match="damaged"):
        load_input(encode(Image.fromarray(noise_rgb()), "TIFF"), "shot.cr2")


def test_unnamed_tiff_photo_still_reaches_pillow():
    assert_contract(load_input(encode(Image.fromarray(noise_rgb()), "TIFF")))


def test_jpeg_named_dng_is_still_read():
    assert_contract(load_input(encode(Image.fromarray(noise_rgb()), "JPEG"), "liar.dng"))


# ---- .npy ----


def test_npy_float_bayer_passes_through():
    array = np.random.default_rng(0).random((64, 64), dtype=np.float32)
    raw = load_input(npy(array), "x.npy")
    assert_contract(raw)
    np.testing.assert_array_equal(raw.bayer, array)


@pytest.mark.parametrize(
    "peak, white",
    [(1023, 1023), (4095, 4095), (16383, 16383), (65535, 65535)],
)
def test_integer_bayer_uses_the_smallest_fitting_bit_depth(peak, white):
    array = np.zeros((64, 64), np.uint16)
    array[0, 0] = peak
    array[0, 1] = peak // 2
    raw = load_input(npy(array), "x.npy")
    assert raw.bayer[0, 0] == 1.0
    assert raw.bayer[0, 1] == pytest.approx((peak // 2) / white, abs=1e-6)


def test_npy_pattern_in_the_filename_is_honoured():
    raw = load_input(npy(mosaic("BGGR")), "scene_bggr.npy")
    assert_rggb_levels(raw, 4095)
    assert raw.meta["cfa"] == "BGGR" and not raw.meta["cfa_assumed"]


def test_npy_without_pattern_is_marked_as_assumed_rggb():
    raw = load_input(npy(mosaic("RGGB")), "scene.npy")
    assert raw.meta["cfa"] == "RGGB" and raw.meta["cfa_assumed"]


def test_npy_rgb_is_unprocessed():
    raw = load_input(npy(noise_rgb()), "x.npy")
    assert_contract(raw)
    assert raw.bayer.shape == (64, 64)


def test_npy_pickle_payload_is_not_executed(tmp_path):
    marker = tmp_path / "pwned"
    payload = np.array([Exploit(marker)], dtype=object)
    buf = io.BytesIO()
    np.save(buf, payload, allow_pickle=True)
    with pytest.raises(InputError):
        load_input(buf.getvalue(), "x.npy")
    assert not marker.exists()


class Exploit:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return (open, (self.path, "w"))


@pytest.mark.parametrize(
    "array",
    [
        np.full((64, 64), np.nan, np.float32),
        np.full((64, 64), np.inf, np.float32),
        np.full((64, 64), 2.5, np.float32),
        np.full((64, 64), -0.5, np.float32),
        np.full((64, 64), -5, np.int32),
        np.ones((64, 64), bool),
        np.ones((64, 64), np.complex64),
        np.zeros(64, np.float32),
        np.zeros((2, 64, 64, 3), np.float32),
        np.zeros((64, 64, 5), np.float32),
    ],
    ids=["nan", "inf", "over", "negative", "neg-int", "bool", "complex", "1d", "4d", "5ch"],
)
def test_bad_npy_arrays_are_refused(array):
    with pytest.raises(InputError):
        load_input(npy(array), "x.npy")


def test_all_black_bayer_is_accepted():
    raw = load_input(npy(np.zeros((64, 64), np.uint16)), "x.npy")
    assert raw.bayer.max() == 0


# ---- 16-bit PNG Bayer ----


def test_png16_is_treated_as_bayer():
    raw = load_input(png16(mosaic("RGGB")), "bayer.png")
    assert_contract(raw)
    assert_rggb_levels(raw, 4095)


def test_png16_pattern_in_the_filename_is_honoured():
    assert_rggb_levels(load_input(png16(mosaic("GRBG")), "x_grbg.png"), 4095)


# ---- JPEG / PNG photographs ----


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_rgb_photo_is_unprocessed(fmt):
    raw = load_input(encode(Image.fromarray(noise_rgb()), fmt), "photo." + fmt.lower())
    assert_contract(raw)
    assert raw.bayer.shape == (64, 64)
    assert raw.wb_gains is not None and raw.ccm is not None


def test_exif_rotation_is_applied():
    exif = Image.Exif()
    exif[0x0112] = 6
    data = encode(Image.fromarray(noise_rgb(64, 96)), "JPEG", exif=exif)
    assert load_input(data, "phone.jpg").bayer.shape == (96, 64)


@pytest.mark.parametrize("fmt, name", [("TIFF", "a.tif"), ("WEBP", "a.webp"), ("BMP", "a.bmp")])
def test_other_photo_formats_are_read(fmt, name):
    assert_contract(load_input(encode(Image.fromarray(noise_rgb()), fmt), name))


def test_grayscale_jpeg_is_converted():
    raw = load_input(encode(Image.new("L", (64, 64), 128), "JPEG"), "grey.jpg")
    assert_contract(raw)


def test_grayscale_8bit_png_is_a_photo_not_bayer():
    raw = load_input(encode(Image.new("L", (64, 64), 128), "PNG"), "grey.png")
    assert raw.wb_gains is not None


def test_cmyk_jpeg_is_converted():
    raw = load_input(encode(Image.new("CMYK", (64, 64), (0, 100, 100, 0)), "JPEG"), "print.jpg")
    assert_contract(raw)
    assert raw.bayer[0, 0] > raw.bayer[1, 1]  # the red ink stays red


def test_rgba_transparent_pixels_become_white():
    transparent = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    opaque = Image.new("RGB", (64, 64), (255, 255, 255))
    a = load_input(encode(transparent, "PNG"), "logo.png")
    b = load_input(encode(opaque, "PNG"), "white.png")
    np.testing.assert_array_equal(a.bayer, b.bayer)


def test_palette_png_is_converted():
    assert_contract(load_input(encode(Image.new("P", (64, 64), 3), "PNG"), "pal.png"))


@pytest.mark.parametrize("size, want", [((65, 65), (64, 64)), ((67, 64), (64, 66)), ((33, 33), (32, 32))])
def test_odd_sizes_are_cropped_to_even(size, want):
    raw = load_input(encode(Image.new("RGB", size, (9, 90, 200)), "PNG"), "odd.png")
    assert raw.bayer.shape == want


def test_odd_sized_bayer_is_cropped_to_even():
    assert load_input(npy(np.zeros((65, 67), np.uint16)), "x.npy").bayer.shape == (64, 66)


@pytest.mark.parametrize(
    "make",
    [
        lambda: encode(Image.new("RGB", (31, 64)), "PNG"),
        lambda: encode(Image.new("RGB", (64, 31)), "JPEG"),
        lambda: encode(Image.new("RGB", (1, 1)), "PNG"),
        lambda: npy(np.zeros((16, 64), np.uint16)),
        lambda: png16(np.zeros((64, 16))),
        lambda: dng(np.zeros((30, 64), np.uint16)),
    ],
    ids=["png", "jpeg", "1px", "npy", "png16", "dng"],
)
def test_tiny_images_are_refused(make):
    with pytest.raises(InputError, match="small"):
        load_input(make())


def test_dng_too_small_for_libraw_is_still_a_clean_refusal():
    with pytest.raises(InputError):
        load_input(dng(np.zeros((16, 64), np.uint16)))


def test_bggr_shift_that_makes_a_frame_tiny_is_refused():
    with pytest.raises(InputError, match="small"):
        load_input(npy(np.zeros((32, 64), np.uint16)), "x_bggr.npy")


# ---- corrupt ----


def test_truncated_jpeg_is_refused():
    data = encode(Image.fromarray(noise_rgb(256, 256)), "JPEG")
    with pytest.raises(InputError):
        load_input(data[: len(data) // 2], "cut.jpg")


def test_truncated_png_is_refused():
    data = encode(Image.fromarray(noise_rgb(256, 256)), "PNG")
    with pytest.raises(InputError):
        load_input(data[: len(data) // 2], "cut.png")


@pytest.mark.parametrize("name", ["junk.jpg", "junk.png", "junk.dng", "junk.npy", "junk.xyz", None])
def test_random_bytes_are_refused(name):
    with pytest.raises(InputError):
        load_input(np.random.default_rng(0).bytes(4096), name)


def test_error_message_is_friendly():
    with pytest.raises(InputError) as e:
        load_input(b"not an image at all", "holiday.jpg")
    message = str(e.value)
    assert "holiday.jpg" in message
    assert "Traceback" not in message and "OSError" not in message


def test_heic_hint_is_given():
    with pytest.raises(InputError, match="JPEG"):
        load_input(b"\0\0\0\x18ftypheic" + b"\0" * 64, "iphone.heic")


def test_empty_upload_is_refused():
    with pytest.raises(InputError, match="empty"):
        load_input(b"", "nothing.jpg")


def test_missing_path_is_refused(tmp_path):
    with pytest.raises(InputError, match="find"):
        load_input(str(tmp_path / "gone.jpg"))


def test_directory_path_is_refused(tmp_path):
    with pytest.raises(InputError):
        load_input(str(tmp_path))


@pytest.mark.parametrize("bad", [None, 12, ["x"]])
def test_wrong_source_type_is_refused(bad):
    with pytest.raises(InputError):
        load_input(bad)


# ---- huge ----


def header_only_png(w, h):
    """Claims a huge size without any pixels, as a decompression bomb would."""

    def chunk(kind, payload):
        crc = zlib.crc32(kind + payload)
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


@pytest.mark.parametrize("w, h", [(12000, 12000), (60000, 60000)], ids=["144MP", "bomb"])
def test_oversized_image_is_refused_before_decoding(w, h):
    start = time.perf_counter()
    with pytest.raises(InputError, match="too large"):
        load_input(header_only_png(w, h), "huge.png")
    assert time.perf_counter() - start < 1.0


def test_oversized_bayer_is_refused():
    with pytest.raises(InputError, match="too large"):
        load_input(npy(np.zeros((12000, 12000), np.uint8)), "huge.npy")


def test_large_photo_is_downscaled_and_records_it():
    data = encode(Image.new("RGB", (7000, 5000), (120, 130, 140)), "JPEG")
    raw = load_input(data, "big.jpg")
    assert_contract(raw)
    assert raw.bayer.size <= 24e6
    assert raw.meta["downscaled_from"] == (5000, 7000)


def test_large_bayer_is_binned_keeping_rggb():
    raw = load_input(npy(mosaic("RGGB", 5000, 7000)), "big.npy")
    assert_contract(raw)
    assert raw.bayer.size <= 24e6
    assert_rggb_levels(raw, 4095)
    assert raw.meta["downscaled_from"] == (5000, 7000)


def test_12mp_photo_keeps_full_resolution():
    data = encode(Image.new("RGB", (4000, 3000), (120, 130, 140)), "JPEG")
    raw = load_input(data, "phone.jpg")
    assert raw.bayer.shape == (3000, 4000)
    assert "downscaled_from" not in raw.meta
