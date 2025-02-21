from typing import Any, Dict, List, Optional, Tuple, Union

import capybara as cb
import cv2
import numpy as np

from ..face_normalization import transform_pts2d
from ..utils import download_model_and_return_model_fpath
from .utils import calc_distance, get_line_lenght, get_vector, norm_ratio, prepare_input_data


class CoorinateReg:
    FaceEdge = slice(0, 33)
    RightEye = slice(33, 43)
    LeftEye = slice(87, 97)
    RightEyeBrow = slice(43, 52)
    LeftEyeBrow = slice(97, 106)
    Nose = slice(72, 87)
    MouthInner = [65, 54, 60, 57, 69, 70, 62, 66]
    MouthOutter = [52, 55, 56, 53, 59, 58, 61, 68, 67, 63, 64]

    NumLandmarks = 106

    # for mouth scores
    MouthWidth = [52, 61]
    MouthHeight = [53, 71]

    # for eye scores
    EyePoints = {"right": {"width": [35, 39], "height": [33, 40]}, "left": {"width": [81, 93], "height": [87, 94]}}

    file_ids = {
        "coordinate_reg_mb1_fp32": "1rS2gsBvxvNktK9sEYEu3aZhWTT4DBweV",  # This model only supports batch_size=1.
        "coordinate_reg_mb1_int8": "1XpNsPdHJSFFJS7IgPe88j8NssHMVqyTi",  # This model only supports batch_size=1.
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_version: str = "coordinate_reg_mb1_fp32",
        gpu_id: int = 0,
        backend: str = "cuda",
        session_option: Dict[str, Any] = {},
        provider_option: Dict[str, Any] = {},
        mouth_th: Optional[float] = 0.2,
        eye_th: Optional[float] = 0.5,
    ):
        if model_path is None:
            model_path = download_model_and_return_model_fpath(
                file_id=self.file_ids[model_version],
                model_fname=model_version + ".onnx",
            )
        self.model_path = model_path
        self.engine = cb.ONNXEngine(
            self.model_path,
            gpu_id=gpu_id,
            backend=backend,
            session_option=session_option,
            provider_option=provider_option,
        )
        self.metadata = cb.PowerDict(self.engine.metadata)
        self.metadata["InputSize"] = self.engine.input_infos["data"]["shape"][1:]
        self.metadata["InputSize"][0] = 1
        self.metadata["Mouth_th"] = mouth_th
        self.metadata["Eye_th"] = eye_th
        self.metadata.freeze()

    def preprocess(self, imgs: List[np.ndarray], boxes: List[cb.Box]) -> Dict[str, Any]:
        blobs = []
        Ms = []
        for img, box in zip(imgs, boxes):
            blob, M = prepare_input_data(
                img=img,
                box=box,
                out_size=self.metadata["InputSize"][-2:],
                expand_ratio=1.2,
            )
            blobs.append(blob)
            Ms.append(M)
        return blobs, Ms

    def postprocess(
        self,
        preds: np.ndarray,
        Ms: List[np.ndarray],
    ) -> List[np.ndarray]:
        h, w = self.metadata["InputSize"][-2:]
        lmks = []
        for pred, M in zip(preds, Ms):
            pred = pred.reshape(-1, 2)
            pred += 1  # -1 ~ 1 -> 0 ~ 2
            pred *= (h // 2, w // 2)
            IM = cv2.invertAffineTransform(M)
            lmk = transform_pts2d(pred, IM).astype(float)
            lmks.append(lmk)
        return lmks

    @staticmethod
    def _face_norm_degree(lmks) -> np.number:
        is_left = get_line_lenght(lmks[31], lmks[33]) < get_line_lenght(lmks[33], lmks[35])
        nose_mid = get_vector(lmks[27], lmks[33])
        face_mid = get_vector(lmks[27], lmks[8])
        inner = np.inner(nose_mid, face_mid)
        norms = np.linalg.norm(nose_mid) * np.linalg.norm(face_mid)
        cos = inner / norms
        rad = np.arccos(np.clip(cos, -1.0, 1.0))
        if is_left:
            return -np.rad2deg(rad) / 90
        else:
            return np.rad2deg(rad) / 90

    def _calc_mouth_scores(self, lmks: np.ndarray) -> np.number:
        if len(lmks) != self.NumLandmarks:
            raise ValueError(f"Landmarks should be {self.NumLandmarks}, but got {len(lmks)}")
        n_degree = self._face_norm_degree(lmks)
        diff_w = calc_distance(lmks, self.MouthWidth)
        diff_h = calc_distance(lmks, self.MouthHeight)
        return norm_ratio(diff_w, diff_h, 0.4, n_degree)

    def _calc_eye_score(self, lmks: np.ndarray, eye_mode="left") -> Tuple[np.number, np.number]:
        if len(lmks) != self.NumLandmarks:
            raise ValueError(f"Landmarks should be {self.NumLandmarks}, but got {len(lmks)}")
        eye_w = calc_distance(lmks, self.EyePoints[eye_mode]["width"])
        eye_h = calc_distance(lmks, self.EyePoints[eye_mode]["height"])
        eye_ratio = norm_ratio(eye_w, eye_h, 1, 0)
        return eye_ratio

    def __call__(
        self,
        imgs: List[np.ndarray],
        boxes: List[cb.Box],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        blobs, Ms = self.preprocess(imgs, boxes)
        preds = {k: [] for k in self.engine.output_infos.keys()}
        for blob in blobs:
            inputs = {"data": blob}
            tmp_preds = self.engine(**inputs)
            for k, v in tmp_preds.items():
                preds[k].append(v)
        lmks = self.postprocess(preds["fc1"], Ms)

        mouth_score = np.array([self._calc_mouth_scores(x) for x in lmks], dtype=float)
        right_eye_score = np.array([self._calc_eye_score(x, "right") for x in lmks], dtype=float)
        left_eye_score = np.array([self._calc_eye_score(x, "left") for x in lmks], dtype=float)

        outs = [
            {
                "lmk": lmks[i],
                "mouth_score": mouth_score[i],
                "right_eye_score": right_eye_score[i],
                "left_eye_score": left_eye_score[i],
                "info": {
                    "model_fpath": self.model_path,
                    "thresholds": {
                        "eye": self.metadata["Eye_th"],
                        "mouth": self.metadata["Mouth_th"],
                    },
                },
            }
            for i in range(len(lmks))
        ]
        return outs

    @classmethod
    def draw_result(
        cls,
        img: np.ndarray,
        face_box: cb.Box,
        result: Dict[str, np.ndarray],
        plot_details: bool = False,
    ) -> np.ndarray:
        max_text_size = img.shape[0] // 30 + 1
        if plot_details:
            text = (
                f"Mouth: {result['mouth_score'].round(4)}\n"
                f"Right Eye: {result['right_eye_score'].round(4)}\n"
                f"Left Eye: {result['left_eye_score'].round(4)}\n"
            )
            text_size = np.clip(round(face_box.height * 0.05) + 1, 1, max_text_size)
            img = cb.draw_text(
                img=img,
                text=text,
                location=face_box.left_bottom,
                text_size=text_size,
            )

        img = cb.draw_points(img, result["lmk"], scales=1, colors=(0, 255, 0))

        if plot_details:
            detail = {
                "Model Path": result["info"]["model_fpath"],
                "THs": result["info"]["thresholds"],
            }
            text = ""
            for k, v in detail.items():
                text += f"{k}: {v}\n"
            text_size = np.clip(max(img.shape) // 100 + 1, 12, 20)
            location = (5, 5)
            img = cb.draw_text(
                img=img,
                text=text,
                location=location,
                text_size=text_size,
            )

        return img
