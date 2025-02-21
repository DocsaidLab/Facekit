from enum import Enum, unique

import capybara as cb
import numpy as np


def l2_norm(arr, axis=1):
    norm = np.linalg.norm(arr, 2, axis, True)
    output = arr / norm
    return output


def distance(arr1, arr2, order):
    dists = np.power(np.sum(np.power(np.abs(arr1 - arr2), order), axis=-1), 1 / order)
    return dists


def std_distance(arr1, arr2, order=2):
    diffs = arr1 - arr2
    dists = np.linalg.norm(diffs, ord=order, axis=1, keepdims=True)
    return dists


@unique
class Distance(cb.EnumCheckMixin, Enum):
    Degree = 0
    Manhattan = 1
    Euclidean = 2

    @staticmethod
    def compare(arr1: np.ndarray, arr2: np.ndarray, code: "Distance"):
        code = Distance.obj_to_enum(code)
        if code == Distance.Degree:
            return ((arr1 * arr2).sum(-1) / 2 + 0.5).clip(0, 1)
        return distance(arr1, arr2, code.value)
