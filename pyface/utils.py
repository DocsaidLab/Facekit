import capybara as cb
import onnxruntime as ort

from .components.face_depth.tddfav2 import TDDFAV2
from .components.face_detection.scrfd import SCRFD
from .components.face_gender import GenderDetector
from .components.face_landmark.coordinate_reg import CoordinateReg
from .components.face_recognition.arcface import ArcFace

__all__ = [
    "download_models",
    "get_ort_backend",
]


def download_models():
    TDDFAV2.download_models()
    SCRFD.download_models()
    GenderDetector.download_models()
    CoordinateReg.download_models()
    ArcFace.download_models()


def cuda_avaliable():
    providers = ort.get_available_providers()
    return "CUDAExecutionProvider" in providers


def coreml_avaliable():
    providers = ort.get_available_providers()
    return "CoreMLExecutionProvider" in providers


def get_ort_backend():
    if cuda_avaliable():
        return cb.Backend.cuda
    elif coreml_avaliable():
        return cb.Backend.coreml
    else:
        return cb.Backend.cpu
