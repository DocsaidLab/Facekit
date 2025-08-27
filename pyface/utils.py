from .components.face_depth.tddfav2 import TDDFAV2
from .components.face_detection.scrfd import SCRFD
from .components.face_gender import GenderDetector
from .components.face_landmark.coordinate_reg import CoordinateReg
from .components.face_recognition.arcface import ArcFace

__all__ = [
    "download_models",
]


def download_models():
    TDDFAV2.download_models()
    SCRFD.download_models()
    GenderDetector.download_models()
    CoordinateReg.download_models()
    ArcFace.download_models()
