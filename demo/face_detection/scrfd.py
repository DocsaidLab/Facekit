from typing import Tuple

import capybara as cb
from fire import Fire

import facekit as fk


def main(
    img_path: str,
    score_th: float = 0.5,
    inp_size: Tuple[int, int] = (480, 640),
):
    cur_folder = cb.get_curdir(__file__)
    face_detection = fk.SCRFD(
        batch_size=8,
        score_th=score_th,
        inp_size=inp_size,
    )
    img = cb.imread(img_path)
    proposals_list = face_detection([img] * 7)
    plotted = face_detection.draw_proposals(img, proposals_list[0])
    cb.imwrite(plotted, cur_folder / "demo.png")


if __name__ == "__main__":
    Fire(main)
