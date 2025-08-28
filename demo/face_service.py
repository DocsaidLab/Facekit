import capybara as cb
from fire import Fire

import pyface as pf

RESOURCE_DIR = cb.get_curdir(__file__).parent / "tests" / "resources"


def main(
    img_path: str = str(RESOURCE_DIR / "EmmaWatson1.jpg"),
    face_bank: str = str(RESOURCE_DIR / "face_bank"),
):
    face_service = pf.FaceService(
        enable_gender=True,
        enable_depth=True,
        enable_recognition=True,
        enable_landmark=True,
        face_bank=face_bank,
    )
    img = cb.imread(img_path)
    faces_on_img = face_service([img], do_1n=True)[0]
    out_folder = cb.get_curdir(__file__) / "output"
    out_folder.mkdir(exist_ok=True, parents=True)
    cb.imwrite(faces_on_img.gen_info_img(), str(out_folder / "face_service.jpg"))


if __name__ == "__main__":
    Fire(main)
