from typing import Any, Dict, List, Optional, Tuple, Union

import capybara as cb
import numpy as np

from ..utils import (
    append_to_batch,
    detach_from_batch,
    download_model_and_return_model_fpath,
)
from .draw import draw_results
from .nms import py_nms


def gen_prior_centers(
    feat_sizes: List[Tuple[int, int]],
    anc_strides: List[Tuple[int, int]],
    anc_img_scales: List[Tuple[int, int]],
):
    prior_centers = []
    for feat_size, stride in zip(feat_sizes, anc_strides):
        xx, yy = np.meshgrid(range(feat_size[1]), range(feat_size[0]))
        prior_grids = np.stack((xx, yy), -1).reshape(-1, 2).repeat(len(anc_img_scales), axis=0)
        prior_grids *= stride
        prior_centers.append(prior_grids)
    prior_centers = np.concatenate(prior_centers, axis=0)
    return prior_centers


def distance2bbox(points, distance) -> np.ndarray:
    bboxes = np.tile(points, 2) + distance * [-1, -1, 1, 1]
    return bboxes


def distance2kps(points, distance) -> np.ndarray:
    kpts = distance + np.tile(points, 5)
    return kpts


def get_flat_preds(preds: Dict[str, np.ndarray], strides: List[int]):
    level_loc_preds = []
    level_obj_preds = []
    level_lmk_preds = []
    for stride in strides:
        box_pred = preds[f"box_{stride}"]  # * stride
        score_pred = preds[f"score_{stride}"]
        lmk_pred = preds[f"lmk5pt_{stride}"]  # * stride

        level_loc_preds.append(box_pred)
        level_obj_preds.append(score_pred)
        level_lmk_preds.append(lmk_pred)

    flat_loc_preds = np.concatenate(level_loc_preds, 1)
    flat_obj_preds = np.concatenate(level_obj_preds, 1)
    flat_lmk_preds = np.concatenate(level_lmk_preds, 1)
    return flat_loc_preds, flat_obj_preds, flat_lmk_preds


def get_proposals(
    flat_preds: Tuple[np.ndarray, np.ndarray, np.ndarray],
    prior_centers: np.ndarray,
    img_scales: List[float],
    score_th: float = 0.02,
    nms_th: float = 0.45,
    nms_topk: int = 5000,
) -> List[Tuple[np.ndarray, ...]]:
    proposals_list = []
    for loc, obj, lmk, img_scale in zip(*flat_preds, img_scales):
        loc = distance2bbox(prior_centers, loc)
        lmk = distance2kps(prior_centers, lmk)

        valid = np.where(obj > score_th)[0]
        loc = loc[valid]
        obj = obj[valid]
        lmk = lmk[valid]

        if len(loc):
            dets = np.concatenate((loc, obj), axis=-1).astype("float32")
            keep = py_nms(dets, thresh=nms_th)[:nms_topk]

            loc = loc[keep]
            obj = obj[keep]
            lmk = lmk[keep].reshape(-1, 5, 2)

        proposals_list.append({
            "boxes": loc / img_scale,
            "scores": obj,
            "lmk5pts": lmk / img_scale,
        })
    return proposals_list


class SCRFD:
    repo_ids = {
        "scrfd_10g_gnkps_fp32": "kunkunlin1221/face-detection_scrfd-10g-gnkps",
        # "scrfd_2.5g_bnkps_fp32": "",
        # "scrfd_34g_gnkps_fp32": "",
    }

    def __init__(
        self,
        model_path: str = None,
        model_name: str = "scrfd_10g_gnkps_fp32",
        batch_size: int = 1,
        inp_size: Optional[Tuple[int, int]] = (480, 640),  # best settings for the model
        score_th: Optional[float] = None,
        gpu_id: int = 0,
        backend: str = "cuda",
        session_option: Dict[str, Any] = {},
        provider_option: Dict[str, Any] = {},
    ):
        if model_path is None:
            model_path = download_model_and_return_model_fpath(
                repo_id=self.repo_ids[model_name],
                model_fname=f"{model_name}.onnx",
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
        self.metadata = cb.PowerDict(self.engine.metadata)
        self.metadata["InputSize"][0] = batch_size
        if inp_size is not None:
            self.metadata["InputSize"][2:] = inp_size
        if score_th is not None:
            self.metadata["ScoreTH"] = score_th
        self.metadata.freeze()
        self.initialize()

    def initialize(self) -> None:
        dummy_inputs = {
            k: np.zeros(self.metadata["InputSize"], dtype=v["dtype"]) for k, v in self.engine.input_infos.items()
        }
        self.engine(**dummy_inputs)

    def preprocess(self, imgs: List[np.ndarray]) -> Union[List[np.ndarray], List[float]]:
        h, w = self.metadata["InputSize"][2:]
        blobs, img_scales = [], []
        for img in imgs:
            blob, img_scale = cb.imresize_and_pad_if_need(
                img=img,
                max_h=h,
                max_w=w,
                return_scale=True,
            )
            if self.metadata["ColorMode"] == "rgb":
                blob = cb.imcvtcolor(blob, "BGR2RGB")
            blob = (blob - self.metadata["Mean"]) / self.metadata["Std"]
            blob = blob.transpose(2, 0, 1)[None].astype("float32")
            blobs.append(blob)
            img_scales.append(img_scale)
        return blobs, img_scales

    def postprocess(self, preds: Dict[str, np.ndarray], img_scales: List[float]) -> List[dict]:
        strides = [x for x in self.metadata["AncStrides"]]
        flat_preds = get_flat_preds(preds, strides)
        h, w = self.metadata["InputSize"][2:]
        if getattr(self, "prior_centers", None) is None:
            feat_sizes = [(int(h / s), int(w / s)) for s in self.metadata["AncStrides"]]
            self.prior_centers = gen_prior_centers(
                feat_sizes,
                self.metadata["AncStrides"],
                self.metadata["AncScales"],
            )
        proposals_list = get_proposals(
            flat_preds=flat_preds,
            prior_centers=self.prior_centers,
            img_scales=img_scales,
            score_th=self.metadata["ScoreTH"],
            nms_th=self.metadata["NMSTH"],
        )
        return [
            {
                "boxes": proposals["boxes"],
                "scores": proposals["scores"],
                "lmk5pts": proposals["lmk5pts"],
                "infos": {
                    "num_proposals": len(proposals["boxes"]),
                    "model_fpath": self.model_path,
                    "thresholds": {"score_th": self.metadata["ScoreTH"]},
                },
            }
            for proposals in proposals_list
        ]

    def __call__(self, imgs: List[np.ndarray]) -> List[dict]:
        blobs, scales = self.preprocess(imgs)
        preds = {k: [] for k in self.engine.output_infos.keys()}
        b = self.metadata["InputSize"][0]
        for batch in cb.make_batch(blobs, b):
            inputs = {name: np.concatenate(append_to_batch(batch, b)) for name, _ in self.engine.input_infos.items()}
            tmp_preds = self.engine(**inputs)
            for k, v in tmp_preds.items():
                preds[k].append(detach_from_batch(v, len(batch)))
        preds = {k: np.concatenate(v, 0) for k, v in preds.items()}
        proposals_list = self.postprocess(preds, scales)
        return proposals_list

    @staticmethod
    def draw_proposals(
        img: np.ndarray,
        proposals: Dict[str, Any],
    ):
        return draw_results(
            img,
            cb.Boxes(proposals["boxes"]),
            proposals["scores"],
            cb.KeypointsList(proposals["lmk5pts"]),
            show_score=True,
        )

    @staticmethod
    def draw_proposals_list(
        imgs: np.ndarray,
        proposals_list: List[Dict[str, Any]],
    ):
        plotteds = []
        for img, proposals in zip(imgs, proposals_list):
            plotteds.append(
                SCRFD.draw_proposals(
                    img,
                    proposals,
                )
            )
        return plotteds

    @classmethod
    def download_models(cls):
        for name, model in cls.repo_ids.items():
            download_model_and_return_model_fpath(repo_id=model, model_fname=f"{name}.onnx")
