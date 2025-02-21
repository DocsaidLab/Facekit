from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import capybara as cb
import numpy as np

from ..utils import get_mapped_score
from .utils import Distance


class RecogLevel(cb.EnumCheckMixin, Enum):
    VeryHigh = 0.9
    High = 0.7
    Medium = 0.5
    Low = 0.3
    VeryLow = 0.1


class FaceCompare:
    def __init__(
        self,
        mapping_table: Dict[str, List[Tuple[float, float]]],
        method: Union[str, Distance] = Distance.Degree,
        recog_level: Union[str, RecogLevel] = RecogLevel.High,
    ):
        self.mapping_table = mapping_table
        self.method = Distance.obj_to_enum(method)
        self.recog_level = RecogLevel.obj_to_enum(recog_level)

    def __call__(
        self,
        enc1: np.ndarray,
        enc2: np.ndarray,
        custom_method: Optional[Union[str, Distance]] = None,
        custom_recog_level: Optional[Union[str, int, RecogLevel]] = None,
        custom_mapping_table: Optional[List[List[float]]] = None,
    ) -> Tuple[float, bool]:
        if enc1.ndim > 3:
            raise ValueError("enc1 should be 1D or 2D array")
        if enc2.ndim > 3:
            raise ValueError("enc2 should be 1D or 2D array")
        if enc1.ndim != enc2.ndim:
            raise ValueError("enc1 and enc2 should have the same shape")

        method = custom_method if custom_method is not None else self.method
        mapping_table = custom_mapping_table if custom_mapping_table is not None else self.mapping_table
        recog_level = custom_recog_level if custom_recog_level is not None else self.recog_level
        threshold = RecogLevel.obj_to_enum(recog_level).value
        sim = Distance.compare(enc1, enc2, method)
        sim = get_mapped_score(sim, mapping_table)
        sim = round(sim, 5)
        return sim, sim >= threshold

    def __repr__(self):
        s = " " * 4
        return f"FaceCompare(\n{s}method={self.method}, \n{s}recog_level={self.recog_level}, \n{s}mapping_table={self.mapping_table},\n)"

    @property
    def recognition_level(self):
        return self.recog_level

    @property
    def compare_method(self):
        return self.method
