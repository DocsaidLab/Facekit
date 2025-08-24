import capybara as cb
from fire import Fire

import pyface.components as pf

cur_folder = cb.get_curdir(__file__)


def main(img_path: str = str(cur_folder / "data" / "EmmaWatson1.jpg")):
    face_detect = pf.build_face_detection()
    face_gender = pf.build_gender_detection()

    img = cb.imread(img_path)
    proposals = face_detect([img])[0]
    face_box = cb.Box(proposals["boxes"][0])
    results = face_gender([img], [face_box])
    plotted = face_gender.draw_results(img, [face_box], results)

    out_folder = cur_folder / "output"
    out_folder.mkdir(exist_ok=True, parents=True)
    cb.imwrite(plotted, out_folder / "face_gender.png")


if __name__ == "__main__":
    Fire(main)
