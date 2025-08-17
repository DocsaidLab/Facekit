from typing import Any, Dict, List, Optional, Tuple, Union

import capybara as cb
import numpy as np

from .utils import (
    append_to_batch,
    detach_from_batch,
    download_model_and_return_model_fpath,
)

__all__ = ["build_gender_detection", "GenderDetector"]


def imresize_and_pad_if_need(
    img: np.ndarray,
    max_h: int,
    max_w: int,
    interpolation: Union[str, int, cb.INTER] = cb.INTER.BILINEAR,
    pad_value: Optional[Union[int, Tuple[int, int, int]]] = 0,
    pad_mode: Union[str, int, cb.BORDER] = cb.BORDER.CONSTANT,
    return_scale: bool = False,
):
    raw_h, raw_w = img.shape[:2]
    scale = min(max_h / raw_h, max_w / raw_w)
    dst_h, dst_w = min(int(raw_h * scale), max_h), min(int(raw_w * scale), max_w)
    img = cb.imresize(
        img,
        (dst_h, dst_w),
        interpolation=interpolation,
    )
    img_h, img_w = img.shape[:2]

    pad_w = max_w - img_w
    pad_h = max_h - img_h
    pad_size = (pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2)
    img = cb.pad(img, pad_size, pad_value, pad_mode)
    if return_scale:
        return img, scale
    else:
        return img


class GenderDetector:
    repo_ids = {
        "gender_detection_lcnet_050": "kunkunlin1221/face-gender-lcnet-050",
    }

    def __init__(
        self,
        model_path: str = None,
        model_version: str = "gender_detection_lcnet_050",
        batch_size: int = 1,
        gpu_id: int = 0,
        backend: str = "cuda",
        session_option: Dict[str, Any] = {},
        provider_option: Dict[str, Any] = {},
    ):
        if model_path is None:
            model_path = download_model_and_return_model_fpath(
                repo_id=self.repo_ids[model_version],
                model_fname=f"{model_version}.onnx",
            )
        self.model_path = model_path
        self.engine = cb.ONNXEngine(
            self.model_path,
            gpu_id=gpu_id,
            backend=backend,
            session_option=session_option,
            provider_option=provider_option,
        )
        print(self.engine)
        self.metadata = self.engine.metadata
        self.metadata["Mean"] = np.array([0.485, 0.456, 0.406], dtype="float32")
        self.metadata["Std"] = np.array([0.229, 0.224, 0.225], dtype="float32")
        self.metadata["InputSize"] = (batch_size, 3, 112, 112)
        self.initialize()

    def initialize(self) -> None:
        dummy_inputs = {"input": np.zeros(self.metadata["InputSize"], dtype="float32")}
        self.engine(**dummy_inputs)

    def preprocess(self, imgs: List[np.ndarray], boxes: List[cb.Box]) -> Dict[str, Any]:
        blobs = []
        for img, box in zip(imgs, boxes):
            x = cb.imcropbox(img, box)
            blob = imresize_and_pad_if_need(
                img=x,
                max_h=112,
                max_w=112,
                return_scale=False,
            )
            blob = blob.astype("float32") / 255
            blob = (blob - self.metadata["Mean"]) / self.metadata["Std"]
            blob = blob.transpose(2, 0, 1)[None]
            blobs.append(blob)
        return blobs

    def postprocess(self, preds: Dict[str, np.ndarray]) -> List[dict]:
        return [
            {
                "gender": "Female" if pred.argmax() else "Male",
                "th": 0.5,
                "info": {
                    "model_fpath": self.model_path,
                },
            }
            for pred in preds["output"]
        ]
        breakpoint()

    def __call__(self, imgs: List[np.ndarray], boxes: List[cb.Box]) -> List[dict]:
        blobs = self.preprocess(imgs, boxes)
        preds = {k: [] for k in self.engine.output_infos.keys()}
        b = self.metadata["InputSize"][0]
        for batch in cb.make_batch(blobs, b):
            inputs = {name: np.concatenate(append_to_batch(batch, b)) for name, _ in self.engine.input_infos.items()}
            tmp_preds = self.engine(**inputs)
            for k, v in tmp_preds.items():
                preds[k].append(detach_from_batch(v, len(batch)))
        preds = {k: np.concatenate(v, 0) for k, v in preds.items()}
        preds = self.postprocess(preds)
        return preds

    @classmethod
    def draw_result(
        cls,
        img: np.ndarray,
        face_box: cb.Box,
        result: Dict[str, np.ndarray],
    ) -> np.ndarray:
        text_size = img.shape[0] // 30 + 1
        img = cb.draw_text(
            img=img,
            text=result["gender"],
            location=face_box.left_bottom,
            text_size=text_size,
            color=(0, 255, 0),
        )

        return img

    def draw_results(
        self,
        img: np.ndarray,
        face_boxes: List[cb.Box],
        results: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        for box, result in zip(face_boxes, results):
            img = self.draw_result(img, box, result)
        return img


def build_gender_detection(name: str = "gender_detection", **kwargs):
    if name == "gender_detection":
        return GenderDetector(**kwargs)
    else:
        raise ValueError(f"Unsupported face recognition model: {name}")
