from enum import Enum

import capybara as cb

__all__ = ["MouthStatus", "EyeStatus", "FacePose"]


class MouthStatus(cb.EnumCheckMixin, Enum):
    Close = 0
    Open = 1
    Smile = 2
    Masked = 3


class EyeStatus(cb.EnumCheckMixin, Enum):
    Close = 0
    Open = 1
    Wink = 2
    Masked = 3


class FacePose(cb.EnumCheckMixin, Enum):
    LeftProfile = 0
    LeftFrontal = 1
    Frontal = 2
    RightFrontal = 3
    RightProfile = 4
    UpFrontal = 5
    DownFrontal = 6
    Unknown = -1
