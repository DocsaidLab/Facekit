from typing import Tuple

import capybara as cb
from fire import Fire

import facekit.components as fk

cur_folder = cb.get_curdir(__file__)


def main(
    img_path: str = str(cur_folder / "data" / "EmmaWatson1.jpg"),
    scale: float = 1,
    dst_size: Tuple[int, int] = (224, 224),
):
    face_detection = fk.SCRFD()
    face_normaliation = fk.FaceNormalize(
        dst_size=dst_size,
        interpolation=cb.INTER.BILINEAR,
        scale=scale,
    )
    img = cb.imread(img_path)
    proposals = face_detection([img])[0]
    norm_img = face_normaliation([img], [cb.Keypoints(proposals["lmk5pts"][0])])[0]
    norm_img = cb.draw_keypoints(
        norm_img,
        face_normaliation.destination_pts,
        scale=1,
    )

    out_folder = cur_folder / "output"
    out_folder.mkdir(exist_ok=True, parents=True)
    cb.imwrite(norm_img, out_folder / f"face_normalize_s={scale}.png")


if __name__ == "__main__":
    Fire(main)
