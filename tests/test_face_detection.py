import capybara as cb
import cv2
import numpy as np
import pytest

import pyface as pf

RESOURCE_DIR = cb.get_curdir(__file__) / "resources"
ANSWER_DIR = RESOURCE_DIR / "answer"
ANSWER_DIR.mkdir(exist_ok=True, parents=True)

TEST_DATA = [
    {
        "img_fpath": RESOURCE_DIR / "EmmaWatson1.png",
        "expected": ANSWER_DIR / "EmmaWatson1_detection.npz",
    },
    {
        "img_fpath": RESOURCE_DIR / "JohnnyDepp1.png",
        "expected": ANSWER_DIR / "JohnnyDepp1_detection.npz",
    },
]


@pytest.mark.parametrize("data", TEST_DATA)
def test_face_detection(data):
    m = pf.build_face_detection(
        name="scrfd",
        model_name="scrfd_10g_gnkps_fp32",
        backend=cb.get_recommended_backend(),
    )
    img = cv2.imread(data["img_fpath"])
    expected = np.load(data["expected"])
    faces_on_img = m(imgs=[img])[0]

    assert faces_on_img["infos"]["num_proposals"]
    np.testing.assert_allclose(faces_on_img["boxes"], expected["boxes"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["lmk5pts"], expected["lmk5pts"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["scores"], expected["scores"], rtol=1e-4)


@pytest.mark.parametrize("data", TEST_DATA)
def test_face_detection_cpu(data):
    m = pf.build_face_detection(name="scrfd", model_name="scrfd_10g_gnkps_fp32", backend="cpu")
    img = cv2.imread(data["img_fpath"])
    expected = np.load(data["expected"])
    faces_on_img = m(imgs=[img] * 8)[0]

    assert faces_on_img["infos"]["num_proposals"]
    np.testing.assert_allclose(faces_on_img["boxes"], expected["boxes"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["lmk5pts"], expected["lmk5pts"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["scores"], expected["scores"], rtol=1e-4)


@pytest.mark.parametrize("data", TEST_DATA)
def test_face_detection_batch(data):
    m = pf.build_face_detection(
        name="scrfd",
        model_name="scrfd_10g_gnkps_fp32",
        batch_size=4,
        backend=cb.get_recommended_backend(),
    )
    img = cv2.imread(data["img_fpath"])
    expected = np.load(data["expected"])
    faces_list = m(imgs=[img] * 5)
    for faces in faces_list:
        assert faces["infos"]["num_proposals"]
        np.testing.assert_allclose(faces["boxes"], expected["boxes"], rtol=1e-4)
        np.testing.assert_allclose(faces["lmk5pts"], expected["lmk5pts"], rtol=1e-4)
        np.testing.assert_allclose(faces["scores"], expected["scores"], rtol=1e-4)


def gen_target():
    m = pf.build_face_detection(name="scrfd", model_name="scrfd_10g_gnkps_fp32", backend="cuda")
    for data in TEST_DATA:
        img = cb.imread(data["img_fpath"])
        faces_on_img = m(imgs=[img] * 8)[0]
        np.savez_compressed(
            data["expected"],
            boxes=faces_on_img["boxes"],
            lmk5pts=faces_on_img["lmk5pts"],
            scores=faces_on_img["scores"],
        )


if __name__ == "__main__":
    gen_target()
