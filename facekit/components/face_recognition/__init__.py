from .arcface import ArcFace
from .compare import FaceCompare, RecogLevel
from .utils import Distance

__all__ = [
    "build_face_recognition",
    "FaceCompare",
    "RecogLevel",
    "Distance",
]

methods = {
    "arcface": ArcFace,
}


def build_face_recognition(name: str = "arcface", **kwargs):
    if name == "arcface":
        return ArcFace(**kwargs)
    else:
        raise ValueError(f"Unsupported face recognition model: {name}")
