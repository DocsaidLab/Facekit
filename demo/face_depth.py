import capybara as cb
from fire import Fire

import facekit.components as fk

cur_folder = cb.get_curdir(__file__)


def main(img_path: str = str(cur_folder / "data" / "EmmaWatson1.jpg")):
    face_detect = fk.build_face_detection()
    face_depth = fk.build_face_depth()

    img = cb.imread(img_path)
    proposals = face_detect([img])[0]
    face_box = cb.Box(proposals["boxes"][0])
    results = face_depth([img], [face_box], return_depth=True)
    plotted = face_depth.draw_results(img, [face_box], results)

    out_folder = cur_folder / "output"
    out_folder.mkdir(exist_ok=True, parents=True)
    cb.imwrite(plotted, out_folder / "face_depth1.png")
    cb.imwrite(results[0]["depth_img"], out_folder / "face_depth2.png")


if __name__ == "__main__":
    Fire(main)
