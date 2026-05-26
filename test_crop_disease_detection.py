import io

from PIL import Image

from backend.routers.platform import _analyse_crop_disease_locally, _normalise_disease_key


def _make_solid_image(color):
    image = Image.new("RGB", (64, 64), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_normalise_disease_aliases_leaf_blight():
    assert _normalise_disease_key("leaf blight") == "early_blight"


def test_local_crop_disease_analysis_returns_structured_result():
    image_bytes = _make_solid_image((48, 160, 48))

    result = _analyse_crop_disease_locally(image_bytes, crop_type="tomato")

    assert result["method"] == "local-heuristic"
    assert result["disease"]
    assert result["confidence"] in {"High", "Medium", "Low"}
    assert 1 <= result["confidenceScore"] <= 99
    assert isinstance(result["treatment"], str)
    assert isinstance(result["prevention"], str)
    assert isinstance(result["pesticides"], list)
    assert isinstance(result["organic"], list)