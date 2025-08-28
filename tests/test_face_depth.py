import capybara as cb
import numpy as np
import pytest

from pyface import build_face_depth
from tests.tools import assert_allclose

RESOURCE_DIR = cb.get_curdir(__file__) / "resources"
ANSWER_DIR = RESOURCE_DIR / "answer"
ANSWER_DIR.mkdir(exist_ok=True, parents=True)

TEST_DATA = [
    {
        "img_fpath": RESOURCE_DIR / "EmmaWatson1.jpg",
        "detect": ANSWER_DIR / "EmmaWatson1_detection.npz",
        "expected": ANSWER_DIR / "EmmaWatson1_depth.npz",
    },
    {
        "img_fpath": RESOURCE_DIR / "JohnnyDepp1.jpg",
        "detect": ANSWER_DIR / "JohnnyDepp1_detection.npz",
        "expected": ANSWER_DIR / "JohnnyDepp1_depth.npz",
    },
]


@pytest.mark.parametrize("data", TEST_DATA)
def test_face_depth(data):
    m = build_face_depth(backend="cpu")
    img = cb.imread(data["img_fpath"])
    expected = np.load(data["expected"], allow_pickle=True)
    boxes = np.load(data["detect"], allow_pickle=True)["boxes"]
    results = m(imgs=[img], boxes=cb.Boxes(boxes), return_depth=True)[0]

    assert_allclose(results["param"], expected["param"])
    assert_allclose(results["lmk3d68pt"], expected["lmk3d68pt"])
    assert_allclose(results["pose_degree"], expected["pose_degree"])
    assert_allclose(results["depth_img"], expected["depth_img"])


@pytest.mark.parametrize("data", TEST_DATA)
def test_face_depth_batch(data):
    m = build_face_depth(batch_size=4)
    img = cb.imread(data["img_fpath"])
    expected = np.load(data["expected"], allow_pickle=True)
    boxes = np.load(data["detect"], allow_pickle=True)["boxes"]
    results = m(imgs=[img] * 6, boxes=cb.Boxes(np.tile(boxes, (6, 1))), return_depth=True)

    for result in results:
        assert_allclose(result["param"], expected["param"])
        assert_allclose(result["lmk3d68pt"], expected["lmk3d68pt"])
        assert_allclose(result["pose_degree"], expected["pose_degree"])
        assert_allclose(result["depth_img"], expected["depth_img"])


def gen_target():
    m = build_face_depth(backend="cpu")
    for data in TEST_DATA:
        img = cb.imread(data["img_fpath"])
        boxes = np.load(data["detect"], allow_pickle=True)["boxes"]
        results = m(imgs=[img], boxes=cb.Boxes(boxes), return_depth=True)[0]
        np.savez_compressed(
            data["expected"],
            param=results["param"],
            lmk3d68pt=results["lmk3d68pt"],
            pose_degree=results["pose_degree"],
            depth_img=results["depth_img"],
            allow_pickle=False,
        )


if __name__ == "__main__":
    gen_target()
