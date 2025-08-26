import capybara as cb
import cv2
import numpy as np
import pytest

import pyface as pf

RESOURCE_DIR = cb.get_curdir(__file__) / "resources"

TEST_DATA = [
    {
        "img_fpath": RESOURCE_DIR / "EmmaWatson1.png",
        "expected": {
            "boxes": np.array([[412.52731323, 194.85551453, 751.04302979, 642.71121216]], dtype="float32"),
            "lmk5pts": np.array(
                [
                    [
                        [449.80097198, 366.0531826],
                        [581.79126072, 367.359972],
                        [465.10058594, 458.07265472],
                        [460.2443161, 527.72718811],
                        [570.59969902, 529.41928101],
                    ]
                ],
                dtype="float32",
            ),
            "scores": np.array([[0.80300564]], dtype="float32"),
        },
    },
    {
        "img_fpath": RESOURCE_DIR / "JohnnyDepp1.png",
        "expected": {
            "boxes": np.array([[434.57501197, 188.672789, 691.22249818, 507.53144979]], dtype="float32"),
            "lmk5pts": np.array(
                [
                    [
                        [478.99239435, 284.38753269],
                        [585.12645214, 287.86553004],
                        [500.60212723, 351.98840269],
                        [480.54181178, 422.01026003],
                        [553.02625371, 426.00003743],
                    ]
                ],
                dtype="float32",
            ),
            "scores": np.array([[0.7277902]], dtype="float32"),
        },
    },
]


@pytest.mark.parametrize("data", TEST_DATA)
def test_build_face_detection(data):
    model = pf.build_face_detection(
        name="scrfd",
        model_name="scrfd_10g_gnkps_fp32",
        backend=pf.get_ort_backend(),
    )
    img = cv2.imread(data["img_fpath"])
    faces_on_img = model(imgs=[img])[0]

    assert faces_on_img["infos"]["num_proposals"]
    np.testing.assert_allclose(faces_on_img["boxes"], data["expected"]["boxes"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["lmk5pts"], data["expected"]["lmk5pts"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["scores"], data["expected"]["scores"], rtol=1e-4)


@pytest.mark.parametrize("data", TEST_DATA)
def test_build_face_detection_cpu(data):
    model = pf.build_face_detection(
        name="scrfd",
        model_name="scrfd_10g_gnkps_fp32",
        backend="cpu",
    )
    img = cv2.imread(data["img_fpath"])
    faces_on_img = model(imgs=[img])[0]

    assert faces_on_img["infos"]["num_proposals"]
    np.testing.assert_allclose(faces_on_img["boxes"], data["expected"]["boxes"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["lmk5pts"], data["expected"]["lmk5pts"], rtol=1e-4)
    np.testing.assert_allclose(faces_on_img["scores"], data["expected"]["scores"], rtol=1e-4)
