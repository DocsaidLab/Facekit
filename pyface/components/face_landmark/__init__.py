from .coordinate_reg import CoordinateReg

__all__ = [
    "build_face_landmark",
]

methods = {
    "coordinate_reg": CoordinateReg,
}


def build_face_landmark(name: str = "coordinate_reg", **kwargs):
    if name in methods:
        return methods[name](**kwargs)
    else:
        raise ValueError(f"Unsupported face landmark model: {name}")
