import capybara as cb
from fire import Fire

import pyface as pf

cur_folder = cb.get_curdir(__file__)


def main(
    img_path: str = str(cur_folder / "data" / "EmmaWatson1.jpg"),
    face_bank: str = str(cur_folder / "data" / "face_bank"),
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
    cb.imwrite(faces_on_img.gen_info_img(), str(cur_folder / "output" / "face_service.jpg"))


if __name__ == "__main__":
    Fire(main)
