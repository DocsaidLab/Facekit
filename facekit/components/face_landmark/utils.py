from typing import Tuple

import capybara as cb
import cv2
import numpy as np
from skimage import transform


def get_vector(p1, p2):
    return p2 - p1


def get_line_lenght(p1, p2):
    return sum((p1 - p2) ** 2) ** 0.5


def norm_ratio(diff_w, diff_h, factor, n_degree):
    diff_w *= factor + n_degree
    ratio = np.linalg.norm(diff_h, axis=-1, ord=2) / np.linalg.norm(diff_w, axis=-1, ord=2)
    return ratio


def calc_distance(lmks, two_points):
    return lmks[two_points[0]] - lmks[two_points[1]]


def calc_mean_distance(lmks, two_points1, two_points2):
    dist1 = lmks[two_points1[0]] - lmks[two_points1[1]]
    dist2 = lmks[two_points2[0]] - lmks[two_points2[1]]
    return np.mean((dist1, dist2), axis=0)


def prepare_input_data(
    img: np.ndarray,
    box: cb.Box,
    out_size: Tuple[int, int],
    expand_ratio: float,
):
    w, h = box.width, box.height
    center = box.center
    rotation = 0
    np_out_size = np.array(out_size)
    scale_ratio = np_out_size / (max(w, h) * expand_ratio)
    t1 = transform.SimilarityTransform(scale=scale_ratio)
    cx, cy = scale_ratio * center
    t2 = transform.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    rot = rotation * np.pi / 180.0
    t3 = transform.SimilarityTransform(rotation=rot)
    t4 = transform.SimilarityTransform(translation=np_out_size / 2)
    t = t1 + t2 + t3 + t4
    M = t.params[:2]
    cropped = cv2.warpAffine(img, M, out_size, borderValue=0.0)
    blob = cv2.dnn.blobFromImage(cropped, 1 / 1, cropped.shape[:2], (0, 0, 0), swapRB=False)
    return blob, M
