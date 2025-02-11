import capybara as cb
from fire import Fire

import facekit as fk


def main(img_path: str):
    cur_folder = cb.get_curdir(__file__)

    face_detect = fk.SCRFD()
    face_depth = fk.TDDFAV2()

    img = cb.imread(img_path)
    proposals = face_detect([img])[0]
    face_box = cb.Box(proposals["boxes"][0])
    results = face_depth([img], [face_box], return_depth=True)
    plotted = face_depth.draw_results(img, [face_box], results)

    cb.imwrite(plotted, cur_folder / "demo.jpg")
    cb.imwrite(results[0]["depth_img"], "demo_depth.jpg")


if __name__ == "__main__":
    Fire(main)
