from .tddfav2 import TDDFAV2

__all__ = [
    "build_face_depth",
]

methods = {
    "tddfav2": TDDFAV2,
}


def build_face_depth(name: str = "tddfav2", **kwargs):
    if name in methods:
        return methods[name](**kwargs)
    else:
        raise ValueError(f"Unsupported face depth model: {name}")
