from .scrfd import SCRFD

__all__ = [
    "build_face_detection",
]

methods = {
    "scrfd": SCRFD,
}


def build_face_detection(name: str = "scrfd", **kwargs):
    if name in methods:
        return methods[name](**kwargs)
    else:
        raise ValueError(f"Unsupported face detection model: {name}")
