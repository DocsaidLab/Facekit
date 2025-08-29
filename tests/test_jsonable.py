import capybara as cb
import pytest

import pyface as pf

RESOURCE_DIR = cb.get_curdir(__file__) / "resources"
ANSWER_DIR = RESOURCE_DIR / "answer"
ANSWER_DIR.mkdir(exist_ok=True, parents=True)

TEST_DATA = [
    {
        "img_fpath": RESOURCE_DIR / "EmmaWatson1.jpg",
        "json_fpath": ANSWER_DIR / "EmmaWatson1_jsonable.json",
    },
    {
        "img_fpath": RESOURCE_DIR / "JohnnyDepp1.jpg",
        "json_fpath": ANSWER_DIR / "JohnnyDepp1_jsonable.json",
    },
]


@pytest.mark.parametrize("data", TEST_DATA)
def test_jsonable(data):
    expected = cb.load_json(data["json_fpath"])
    faces = pf.Faces.from_json(expected)
    output = faces.be_jsonable()
    assert output == expected


def gen_data(data):
    face_service = pf.FaceService(
        enable_depth=True,
        enable_landmark=True,
        enable_recognition=True,
        enable_gender=True,
        face_bank=RESOURCE_DIR / "face_bank",
    )
    img = cb.imread(data["img_fpath"])
    faces = face_service([img], do_1n=True)[0]
    output = faces.be_jsonable()
    cb.dump_json(output, data["expected"])


if __name__ == "__main__":
    for data in TEST_DATA:
        gen_data(data)
