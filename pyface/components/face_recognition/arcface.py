from typing import Any, Dict, List, Optional

import capybara as cb
import numpy as np

from ..face_normalization import FaceNormalize
from ..utils import append_to_batch, detach_from_batch, download_model_and_return_model_fpath
from .utils import l2_norm


class ArcFace:
    repo_ids = {
        "wf42m_pfc03_vit-l_fp32": "kunkunlin1221/face-recognition_vit-l-pfc0.3-cosface-web42m",
    }
    versions = {"wf42m_pfc03_vit-l_fp32": "1.0"}

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "wf42m_pfc03_vit-l_fp32",
        batch_size: int = 1,
        enable_flip: bool = False,
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
        self.metadata = cb.PowerDict(self.engine.metadata)
        self.metadata["InputSize"][0] = batch_size
        self.metadata["Mean"] = [127.5, 127.5, 127.5]
        self.metadata["Std"] = [128, 128, 128]
        self.metadata["ColorMode"] = "rgb"
        self.metadata["Version"] = self.versions[model_name]
        self.metadata.freeze()
        self.initialize()

        self.face_normalizer = FaceNormalize(
            self.metadata["InputSize"][2:],
            fill_value=(0, 0, 0),
            interpolation=cb.INTER.BILINEAR,
            scale=1.0,
            shift=(0, 0),
        )
        self.enable_flip = enable_flip

    def initialize(self) -> None:
        dummy_inputs = {
            k: np.zeros(self.metadata["InputSize"], dtype=v["dtype"]) for k, v in self.engine.input_infos.items()
        }
        self.engine(**dummy_inputs)

    def preprocess(self, imgs: List[np.ndarray], lmk5pts: List[cb.Keypoints]):
        norm_imgs = self.face_normalizer(imgs, lmk5pts)
        blobs = []
        for norm_img in norm_imgs:
            if self.metadata["ColorMode"] != "bgr":
                blob = cb.imcvtcolor(
                    norm_img,
                    cvt_mode=f"BGR2{self.metadata['ColorMode'].upper()}",
                )
            blob = (blob - self.metadata["Mean"]) / self.metadata["Std"]
            blob = blob.transpose(2, 0, 1)[None].astype("float32")
            blobs.append(blob)
        return blobs, norm_imgs

    def __call__(
        self,
        imgs: List[np.ndarray],
        lmk5pts: List[cb.Keypoints],
    ) -> List[Dict[str, Any]]:
        blobs, norm_imgs = self.preprocess(imgs, lmk5pts)
        preds = {k: [] for k in self.engine.output_infos.keys()}
        b = self.metadata["InputSize"][0]
        for batch in cb.make_batch(blobs, b):
            inputs = {name: np.concatenate(append_to_batch(batch, b)) for name, _ in self.engine.input_infos.items()}
            tmp_preds = self.engine(**inputs)
            if self.enable_flip:
                flip_inputs = {k: np.flip(v, 3) for k, v in inputs.items()}
                tmp_flip_preds = self.engine(**flip_inputs)
                for k, v in tmp_preds.items():
                    tmp_preds[k] = np.concatenate([v, tmp_flip_preds[k]], 1)
            for k, v in tmp_preds.items():
                preds[k].append(detach_from_batch(v, len(batch)))
        preds = {k: np.concatenate(v, 0) for k, v in preds.items()}
        emgeddings = l2_norm(preds["encode"])
        return [
            {
                "embeddings": emgeddings[i],
                "norm_img": norm_imgs[i],
                "info": {
                    "model_fpath": self.model_path,
                    "enable_flip": self.enable_flip,
                    "compare_method": self.metadata["CompareMethod"],
                    "version": self.metadata["Version"],
                },
            }
            for i in range(len(imgs))
        ]

    @property
    def mapping_table(self):
        return self.metadata["MScore"]

    @property
    def compare_method(self):
        return self.metadata["CompareMethod"]

    @property
    def version(self):
        return self.metadata["Version"]

    @classmethod
    def download_models(cls):
        for name, model in cls.repo_ids.items():
            download_model_and_return_model_fpath(repo_id=model, model_fname=f"{name}.onnx")
