from .coordinate_reg import CoorinateReg

__all__ = [
    "build_face_landmark",
]

methods = {
    "coorinate_reg": CoorinateReg,
}


def build_face_landmark(name: str = "coorinate_reg", **kwargs):
    if name in methods:
        return methods[name](**kwargs)
    else:
        raise ValueError(f"Unsupported face landmark model: {name}")
