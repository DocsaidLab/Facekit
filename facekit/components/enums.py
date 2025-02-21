from enum import Enum

import capybara as cb

__all__ = [
    "MouthStatus",
    "EyeStatus",
    "FacePose",
    "FakeType",
]


class MouthStatus(cb.EnumCheckMixin, Enum):
    Close = 0
    Open = 1


class EyeStatus(cb.EnumCheckMixin, Enum):
    Close = 0
    Open = 1


class FacePose(cb.EnumCheckMixin, Enum):
    LeftProfile = 0
    LeftFrontal = 1
    Frontal = 2
    RightFrontal = 3
    RightProfile = 4
    UpFrontal = 5
    DownFrontal = 6
    Unknown = -1


class FakeType(cb.EnumCheckMixin, Enum):
    Live = 0
    Print = 1
    Replay = 2
    Mask = 3
    Partial = 4
    Makeup = 5
    Unknown = -1
