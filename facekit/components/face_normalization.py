from random import randint
from typing import List, Tuple, Union

import capybara as cb
import cv2
import numpy as np
from skimage.transform import estimate_transform

__all__ = [
    "get_norm_pts2d",
    "scale_norm_pts2d",
    "transform_pts2d",
    "face_normalize",
    "FaceNormalize",
]


def get_norm_pts2d(dst_size: Tuple[int, int] = (112, 112)):
    reference_pts = np.array(
        [
            [38.2946, 51.6963],  # right eye
            [73.5318, 51.5014],  # left eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # right mouth
            [70.7299, 92.2041],  # left mouth
        ],
        dtype="float32",
    )
    reference_pts *= np.array(dst_size) / 112
    return reference_pts


def scale_norm_pts2d(
    pts: np.ndarray,
    scale: float,
    dst_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    scaled_dst_size = np.array(dst_size) * scale
    shift = scaled_dst_size - dst_size
    scaled_pts = pts * scale - shift / 2
    return scaled_pts


def transform_pts2d(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))], dtype=pts.dtype)
    new_pts_homo = pts_homo @ M.T
    return new_pts_homo


def face_normalize(
    img: np.ndarray,
    lmk5pt: Union[np.ndarray, cb.Keypoints],
    dst_size: Tuple[int, int] = (112, 112),
    fill_value: Union[int, Tuple[int, int, int]] = 0,
    interpolation: Union[int, cb.INTER] = cb.INTER.BILINEAR,
    scale: float = 1.0,
    shift: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """
    This function is used to normalize face.

    Ref: https://github.com/deepinsight/insightface/issues/1286

    Args:
        img (np.ndarray):
            Image to face normalization.
        lmk5pt (cb.Keypoints):
            Origin face 5pt landmarks.
        dst_size (Tuple[int, int], optional):
            The output size of normalized image. Defaults to (112, 112).
        fill_value (Union[int, Tuple[int, int, int]], optional):
            The color value of padding for normalized image. Defaults to 0.
        interpolation (Union[int, cb.Interpolation], optional):
            The interpolation method of normalization. Defaults to cb.Interpolation.BILINEAR.
        scale (float, optional):
            The scale factor of normalized face size. Defaults to 0.
        shift (Tuple[int, int], optional):
            The shift axis of normalized face. Defaults to 0.

    Raises:
        ValueError: The length of face landmarks must be 5.

    Returns:
        np.ndarray: Normalized face image.
    """
    if len(lmk5pt) != 5:
        raise ValueError(f"The length of lmk5pt must be 5, but got {len(lmk5pt)}.")

    src_pts = lmk5pt.numpy() if isinstance(lmk5pt, cb.Keypoints) else lmk5pt
    src_pts = src_pts.astype("float32")

    dst_pts = get_norm_pts2d(dst_size)
    dst_pts = scale_norm_pts2d(dst_pts, scale, dst_size)
    dst_pts = dst_pts + shift
    M = estimate_transform("similarity", src_pts, dst_pts).params[:2]
    warpped = cv2.warpAffine(
        src=img,
        M=M,
        dsize=dst_size,
        flags=cb.INTER.obj_to_enum(interpolation).value,
        borderValue=fill_value,
    )
    return warpped


class FaceNormalize:
    def __init__(
        self,
        dst_size: Tuple[int, int] = (112, 112),
        fill_value: Union[int, Tuple[int, int, int]] = 0,
        interpolation: Union[int, cb.INTER] = -1,
        scale: float = 1.0,
        shift: Tuple[int, int] = (0, 0),
    ) -> None:
        self.dst_size = dst_size
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.scale = scale
        self.shift = shift

    @property
    def destination_pts(self) -> np.ndarray:
        pts = get_norm_pts2d(self.dst_size)
        pts = scale_norm_pts2d(pts, scale=self.scale, dst_size=self.dst_size)
        pts = pts + self.shift
        return pts

    def __call__(self, imgs: List[np.ndarray], lmk5pts: List[cb.Keypoints]) -> List[np.ndarray]:
        if self.interpolation == -1:
            interpolation = cb.INTER.obj_to_enum(randint(0, 4))
        else:
            interpolation = self.interpolation

        norm_imgs = []
        for img, lmk5pt in zip(imgs, lmk5pts):
            norm_img = face_normalize(
                img,
                lmk5pt,
                dst_size=self.dst_size,
                fill_value=self.fill_value,
                interpolation=interpolation,
                scale=self.scale,
                shift=self.shift,
            )
            norm_imgs.append(norm_img)
        return norm_imgs
