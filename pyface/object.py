from dataclasses import dataclass, field, fields
from typing import Any, List, Optional, Tuple, Union

import capybara as cb
import cv2
import numpy as np

from .components.enums import FacePose, FakeType

__all__ = [
    "Eye",
    "WhetherOrNot",
    "TDDFA",
    "Encode",
    "Who",
    "Face",
    "Faces",
    "Liveness",
    "Attribute",
    "sort_face_by_size",
    "drop_too_small_faces",
    # "faces_to_schema",
    "ATTR_NAMES",
]


@dataclass()
class Eye(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    is_open: Optional[bool] = field(default=None)
    score: Optional[float] = field(default=None)


@dataclass()
class Mouth(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    is_open: Optional[bool] = field(default=None)
    score: Optional[float] = field(default=None)


@dataclass()
class WhetherOrNot(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    is_true: Optional[bool] = field(default=None)
    value: Optional[float] = field(default=None)
    threshold: Optional[float] = field(default=None)


@dataclass()
class Liveness(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    is_true: Optional[bool] = field(default=None)
    value: Optional[Union[float, np.number]] = field(default=None)
    threshold: Optional[Union[float, np.number]] = field(default=None)
    fake_type: Optional[FakeType] = field(default=None)


@dataclass()
class TDDFA(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    param: Optional[np.ndarray] = field(default=None)
    lmk68pt: Optional[np.ndarray] = field(default=None)
    depth_img: Optional[np.ndarray] = field(default=None)
    yaw: Optional[float] = field(default=None)
    roll: Optional[float] = field(default=None)
    pitch: Optional[float] = field(default=None)


@dataclass()
class Encode(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    vector: Optional[np.ndarray] = field(default=None)
    version: Optional[str] = field(default=None)


@dataclass()
class Who(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    name: Optional[str] = field(default="?")
    confidence: Optional[float] = field(default=None)
    recognized_level: Optional[int] = field(default=None)


@dataclass()
class Attribute(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    age: Optional[int] = field(default=None)
    gender: Optional[str] = field(default=None)
    race: Optional[str] = field(default=None)
    pose: Optional[FacePose] = field(default=None)
    left_eye: Optional[Eye] = field(default=None)
    right_eye: Optional[Eye] = field(default=None)
    mouth: Optional[Mouth] = field(default=None)


@dataclass()
class Face(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    box: cb.Box
    score: Union[float, np.number] = field(default=1.0)
    lmk5pt: Optional[cb.Keypoints] = field(default=None)
    norm_img: Optional[np.ndarray] = field(default=None)
    tddfa: Optional[TDDFA] = field(default=None)
    encoding: Optional[Encode] = field(default=None)
    who: Optional[Who] = field(default=None)
    lmk106pt: Optional[cb.Keypoints] = field(default=None)
    liveness: Optional[Liveness] = field(default=None)
    attribute: Optional[Attribute] = field(default=None)
    # assign jsonable functions for some fields
    jsonable_func = {
        "vector": lambda x: cb.npy_to_b64str(x) if x is not None else None,
        "norm_img": lambda x: cb.img_to_b64str(x, cb.IMGTYP.PNG) if x is not None else None,
        "depth_img": lambda x: cb.img_to_b64str(x, cb.IMGTYP.PNG) if x is not None else None,
        "param": lambda x: cb.npy_to_b64str(x) if x is not None else None,
    }
    # pose: Optional[FacePose] = field(default=None)
    # blur: Optional[WhetherOrNot] = field(default=None)
    # occlusion: Optional[Occlusion] = field(default=None)
    # lmk68pt: Optional[cb.Keypoints] = field(default=None)
    # analysis_infos: Optional[dict] = field(default=None)


ATTR_NAMES = [f.name for f in fields(Face)]


def _downscale(img, scale: float = 0.02):
    (w, h) = img.shape[:2][::-1]
    downscaled = cv2.resize(img, None, fx=scale, fy=scale, interploation=cv2.INTER_NEAREST)
    upscaled = cv2.resize(downscaled, (w, h), interploation=cv2.INTER_NEAREST)
    return upscaled


@dataclass()
class Faces(cb.DataclassToJsonMixin, cb.DataclassCopyMixin):
    raw_image: Optional[np.ndarray] = field(default=None)
    faces: List[Face] = field(default_factory=list)

    def __getattr__(self, attr: str):
        if attr not in ["faces", "raw_image"] and not (attr.startswith("__") and attr.endswith("__")):
            for base_name in ATTR_NAMES:
                if attr == base_name:
                    return [getattr(x, base_name) for x in self.faces]
            raise ValueError(f"Given key = {attr} is not valid.")
        return super().__getattr__(attr)

    def __setattr__(self, attr: str, value: List[Any]):
        if attr not in ["faces", "raw_image"]:
            for base_name in ATTR_NAMES:
                if attr == base_name:
                    if not isinstance(value, list):
                        raise ValueError("Given value is not a List, which is not invalid.")
                    if len(value) != len(self.faces):
                        raise ValueError(f"Length error from given value = {len(value)}")
                    return [setattr(x, base_name, v) for x, v in zip(self.faces, value)]
            raise ValueError(f"Given key = {attr} is not valid.")
        return super().__setattr__(attr, value)

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, ind) -> Union["Faces", Face]:
        if isinstance(ind, list):
            faces = [self.faces[i] for i in ind]
            return Faces(self.raw_image, faces=faces)
        elif isinstance(ind, slice):
            faces = self.faces[ind]
            return Faces(self.raw_image, faces=faces)
        elif isinstance(ind, int):
            if isinstance(ind, int) and ind >= len(self):
                raise IndexError("Given ind is out of length of faces.")
            return self.faces[ind]
        else:
            raise IndexError(f"Unsupport type of ind = {type(ind)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def gen_info_img(self, mosaic_face: bool = False):
        if not cb.is_numpy_img(self.raw_image):
            raise ValueError("Given raw_image is not a numpy image.")

        img = self.raw_image.copy()

        if len(self.faces):
            zipped = zip(
                self.box,
                self.score,
                self.lmk5pt,
                self.attribute,
                self.tddfa,
                self.who,
                self.lmk106pt,
                self.liveness,
            )
            for box, score, lmk5pt, attribute, tddfa, who, lmk106pt, liveness in zipped:
                text_size = np.clip(round(box.height / 5), 8, 32)
                box_line_scale = (box.width / 128).clip(1, 3)
                point_scale = (box.width / 256).clip(0.3, 2)

                # 馬賽克
                if mosaic_face:
                    (x1, y1), (x2, y2) = round(box.left_top), round(box.right_bottom)
                    img[y1:y2, x1:x2] = _downscale(img[y1:y2, x1:x2])

                img = cb.draw_box(img, box, thickness=box_line_scale)
                text = f"Score: {round(score[0], 4):.4f}"

                loc = box.left_top - (0, text_size + box.height // 50 + 1)
                img = cb.draw_text(
                    img,
                    text,
                    location=loc,
                    color=(0, 0, 0),
                    text_size=text_size,
                    stroke_width=2,
                    stroke_fill=(100, 255, 100),
                )

                loc = box.left_bottom
                text_to_draw = ""

                if isinstance(attribute, Attribute):
                    if attribute.gender is not None:
                        text_to_draw += f"Gender: {attribute.gender}\n"
                    if attribute.age is not None:
                        text_to_draw += f"Age: {attribute.age}\n"
                    if attribute.race is not None:
                        text_to_draw += f"Race: {attribute.race}\n"
                    if isinstance(attribute.pose, FacePose):
                        text_to_draw += f"Pose: {attribute.pose.name}\n"
                    if isinstance(attribute.right_eye, Eye):
                        text_to_draw += f"REye: {'open' if attribute.right_eye.is_open else 'close'} "
                    if isinstance(attribute.left_eye, Eye):
                        text_to_draw += f"LEye: {'open' if attribute.left_eye.is_open else 'close'} "
                    if isinstance(attribute.mouth, Mouth):
                        text_to_draw += f"Mouth: {'open' if attribute.mouth.is_open else 'close'}\n"

                if isinstance(who, Who):
                    text_to_draw += f"Who: {who.name}\n"

                if isinstance(liveness, Liveness):
                    text_to_draw += f"FAS: like {liveness.label}\n"

                if isinstance(tddfa, TDDFA):
                    text_to_draw += f"Yaw: {tddfa.yaw:.2f}, Roll: {tddfa.roll:.2f}, Pitch: {tddfa.pitch:.2f}\n"

                if lmk106pt is not None:
                    img = cb.draw_points(img, lmk106pt.numpy(), point_scale, colors=(100, 220, 0))

                img = cb.draw_keypoints(img, lmk5pt, point_scale)

                img = cb.draw_text(
                    img,
                    text_to_draw,
                    location=box.left_bottom + (0, box.height // 50 + 1),
                    color=(0, 0, 0),
                    text_size=text_size // 1.5,
                    stroke_width=2,
                    stroke_fill=(100, 255, 100),
                )

        return img

    def be_jsonable(self):
        raw_image = cb.img_to_b64str(self.raw_image, cb.IMGTYP.PNG) if self.raw_image is not None else None
        return {
            "raw_image": raw_image,
            "faces": [x.be_jsonable() for x in self.faces],
        }


# def _remove_none_in_jsonized_face(jsonized_face: dict) -> dict:
#     outs = {}
#     for k, v in jsonized_face.items():
#         if v is None:
#             continue

#         if isinstance(v, dict):
#             outs[k] = _remove_none_in_jsonized_face(v)
#         else:
#             if k == "box":
#                 outs[k] = {"leftTop": v[:2], "rightBottom": v[2:]}
#             else:
#                 outs[k] = v
#     return outs


# def faces_to_schema(faces: Faces):
#     jsonized_faces = faces.be_jsonable()["faces"]
#     jsonized_faces = [_remove_none_in_jsonized_face(x) for x in jsonized_faces]
#     return {"faces": jsonized_faces}


def sort_face_by_size(faces_list: List[Faces]):
    for i, faces in enumerate(faces_list):
        if len(faces):
            areas = cb.Boxes(faces.box).area
            inds = np.argsort(areas)[::-1].tolist()
            faces_list[i] = faces_list[i][inds]
    return faces_list


def drop_too_small_faces(
    faces_list: List[Faces],
    small_size: Tuple[int, int],
) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    h, w = small_size
    small_imgs = []
    small_inds = []
    small_shifts = []
    for i, faces in enumerate(faces_list):
        if len(faces):
            boxes = cb.Boxes(faces.box)
            small_mask = np.stack((boxes.width < w, boxes.height < h)).any(0)
            if small_mask.sum():
                small_boxes = boxes[small_mask].scale(fx=2, fy=2)
                cropped_imgs = cb.imcropboxes(faces.raw_image, small_boxes, use_pad=True)
                cropped_imgs = [cb.imadjust(x) for x in cropped_imgs]
                small_imgs.extend(cropped_imgs)
                small_inds.extend([(i, j) for j in np.where(small_mask)[0].tolist()])
                small_shifts.extend(small_boxes.left_top)
    return small_imgs, small_inds, small_shifts
