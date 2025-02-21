import capybara as cb
from fire import Fire

import facekit as fk

cur_folder = cb.get_curdir(__file__)


def main(
    img_path: str = str(cur_folder / "data" / "EmmaWatson1.jpg"),
    face_bank: str = str(cur_folder / "data" / "face_bank"),
):
    face_service = fk.FaceService(
        enable_depth=True,
        enable_recognition=True,
        enable_landmark=True,
        face_bank=face_bank,
    )
    img = cb.imread(img_path)
    face = face_service([img], do_1n=True)[0][0]
    print(f"Recognize as {face.who.name} with confidence score = {face.who.confidence:.5f} (0-1)")
    print(f"Recognize level: {face.who.recognized_level}")


if __name__ == "__main__":
    Fire(main)
