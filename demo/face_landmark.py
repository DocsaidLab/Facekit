from typing import Tuple

import capybara as cb
from fire import Fire

import pyface as pf

cur_folder = cb.get_curdir(__file__)


def main(
    img_path: str = str(cur_folder / "data" / "EmmaWatson1.jpg"),
    score_th: float = 0.5,
    inp_size: Tuple[int, int] = (480, 640),
):
    face_detection = pf.build_face_detection(
        batch_size=8,
        score_th=score_th,
        inp_size=inp_size,
    )
    face_landmark = pf.build_face_landmark()
    img = cb.imread(img_path)
    proposals = face_detection([img])[0]
    box = cb.Box(proposals["boxes"][0])
    result = face_landmark([img], [box])[0]
    plotted = face_landmark.draw_result(img, box, result, plot_details=True)

    out_folder = cur_folder / "output"
    out_folder.mkdir(exist_ok=True, parents=True)
    cb.imwrite(plotted, out_folder / "face_landmark.jpg")


if __name__ == "__main__":
    Fire(main)
