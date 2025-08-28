from typing import Tuple

import capybara as cb
from fire import Fire

import pyface.components as pf

RESOURCE_DIR = cb.get_curdir(__file__).parent / "tests" / "resources"


def main(
    img_path: str = str(RESOURCE_DIR / "EmmaWatson1.jpg"),
    score_th: float = 0.5,
    inp_size: Tuple[int, int] = (480, 640),
):
    face_detection = pf.build_face_detection(
        batch_size=8,
        score_th=score_th,
        inp_size=inp_size,
    )
    img = cb.imread(img_path)
    proposals_list = face_detection([img] * 7)
    plotted = face_detection.draw_proposals(img, proposals_list[0])

    out_folder = cb.get_curdir(__file__) / "output"
    out_folder.mkdir(exist_ok=True, parents=True)
    cb.imwrite(plotted, out_folder / "face_detection.png")


if __name__ == "__main__":
    Fire(main)
