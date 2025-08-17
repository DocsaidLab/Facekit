from math import asin, atan2, cos
from typing import Any, Dict, List, Optional, Tuple, Union

import capybara as cb
import cv2
import numpy as np

from ..enums import FacePose
from ..utils import (
    append_to_batch,
    detach_from_batch,
    download_model_and_return_model_fpath,
)

try:
    import sim3dr_cython as sim3dr
except ImportError:
    import warnings

    warnings.warn("sim3dr_cython not found, make sure you have installed it.")


def rasterize(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray,
    bg: Optional[np.ndarray] = None,
    h: Optional[int] = None,
    w: Optional[int] = None,
    c: Optional[int] = None,
    reverse: bool = False,
):
    if bg is not None:
        h, w, c = bg.shape
    else:
        assert h is not None and w is not None and c is not None
        bg = np.zeros((h, w, c), dtype=np.uint8)

    buffer = np.zeros((h, w), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    sim3dr.rasterize(
        bg,
        vertices,
        triangles,
        colors,
        buffer,
        triangles.shape[0],
        h,
        w,
        c,
        reverse=reverse,
    )
    return bg


def arr_to_ctype(arr: np.ndarray) -> np.ndarray:
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr


def depth(
    img: np.ndarray,
    vertices: np.ndarray,
    tri: np.ndarray,
) -> np.ndarray:
    overlap = img.copy()

    ver = arr_to_ctype(vertices)  # transpose

    z = ver[:, 2]
    z_min, z_max = min(z), max(z)

    z = (z - z_min) / (z_max - z_min + 1e-6)

    # expand
    z = z[:, None].repeat(3, axis=1)

    overlap = rasterize(ver.astype("float32"), tri, z, bg=overlap)

    return overlap


def P2sRt(
    P: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Decompositing camera matrix P.

    Args:
        P (np.ndarray): (3, 4). Affine Camera Matrix.

    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def matrix2angle(R: np.ndarray) -> Tuple[float, float, float]:
    """
    To compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv

    Args:
        R (np.ndarray): (3, 3). rotation matrix

    Returns:
        x: yaw
        y: pitch
        z: roll
    """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z


def calc_pose(param: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = param[:12].reshape(3, -1)  # camera matrix
    _, R, t3d = P2sRt(P)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)
    pose = [p * 180 / np.pi for p in pose]
    return P, pose


def _prepare_tri(tri_path: Optional[Union[str, cb.Path]]) -> np.ndarray:
    tri_array = np.load(tri_path)
    tri_array = arr_to_ctype(tri_array.T).astype("int32")
    return tri_array


def _prepare_bfm_npz(
    bfm_npz_path: Optional[Union[str, cb.Path]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bfm_dict = dict(np.load(bfm_npz_path))
    u = bfm_dict.get("u").astype("float32")
    w_shp = bfm_dict.get("w_shp").astype("float32")[..., :40]
    w_exp = bfm_dict.get("w_exp").astype("float32")[..., :10]
    keypoints = bfm_dict.get("keypoints").astype("long")

    u_base = u[keypoints].reshape(-1, 1)
    w_shp_base = w_shp[keypoints]
    w_exp_base = w_exp[keypoints]
    return u_base, w_shp_base, w_exp_base


def _parse_tffda_param(
    param: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    matrix pose form
    param: shape = (trans_dim + shape_dim + exp_dim) i.e., 62 = 12 + 40 + 10
    """
    P = param[:12].reshape(3, -1).astype("float32")
    alpha_shp = param[12:52].reshape(-1, 1).astype("float32")
    alpha_exp = param[52:-4].reshape(-1, 1).astype("float32")
    scale = param[62:64].astype("float32")
    shift = param[64:66].astype("float32")
    return P, alpha_shp, alpha_exp, scale, shift


class TDDFAV2:
    repo_ids = {
        "tddfav2_mbv1_fp32": "kunkunlin1221/face-depth-3d-68_tddfav2-mbv1",
        # postprocess models
        "bfm_onnx": "kunkunlin1221/face-depth-3d-68_tddfav2-mbv1",
        "bfm_npz": "kunkunlin1221/face-depth-3d-68_tddfav2-mbv1",
        "tri_npy": "kunkunlin1221/face-depth-3d-68_tddfav2-mbv1",
    }

    def __init__(
        self,
        model_path: Optional[Union[str, cb.Path]] = None,
        model_version: str = "tddfav2_mbv1_fp32",
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
        self.batch_size = batch_size
        self.engine = cb.ONNXEngine(
            self.model_path,
            gpu_id=gpu_id,
            backend=backend,
            session_option=session_option,
            provider_option=provider_option,
        )
        print(self.engine, flush=True)
        self.metadata = cb.PowerDict(self.engine.metadata)
        self.metadata.freeze()

        bfm_onnx_path = download_model_and_return_model_fpath(
            repo_id=self.repo_ids["bfm_onnx"],
            model_fname="bfm_noneck_v3.onnx",
        )
        bfm_npz_path = download_model_and_return_model_fpath(
            repo_id=self.repo_ids["bfm_npz"],
            model_fname="bfm_noneck_v3.npz",
        )
        tri_npy_path = download_model_and_return_model_fpath(
            repo_id=self.repo_ids["tri_npy"],
            model_fname="tri.npy",
        )
        self.bfm_engine = cb.ONNXEngine(
            bfm_onnx_path,
            gpu_id=gpu_id,
            backend=backend,
            session_option=session_option,
            provider_option=provider_option,
        )
        self._u, self._w_shp, self._w_exp = _prepare_bfm_npz(bfm_npz_path)
        self._tri = _prepare_tri(tri_npy_path)

    def initialize(self) -> None:
        dummy_inputs = {
            k: np.random.randn(self.batch_size, *v.shape[1:]).astype(v["dtype"])
            for k, v in self.engine.input_infos.items()
        }
        self.engine(**dummy_inputs)

        input_dims = {
            "R": [3, 3],
            "offset": [3, 1],
            "alpha_shp": [40, 1],
            "alpha_exp": [10, 1],
        }
        dummy_inputs = {k: np.zeros(input_dims[k], dtype=v["dtype"]) for k, v in self.bfm_engine.input_infos.items()}
        self.bfm_engine(**dummy_inputs)

    def _transform(
        self,
        img: np.ndarray,
        box: cb.Box,
    ) -> Tuple[np.ndarray, Tuple[float, float], np.ndarray]:
        inp_w, inp_h = self.engine.input_infos["input"]["shape"][2:]

        expanded_box = box.square().scale(fx=0.98, fy=0.98)
        shift = expanded_box.left_top
        scale = expanded_box.width / inp_w, expanded_box.height / inp_h
        blob = cb.imcropbox(img, expanded_box)
        blob = cb.imresize(blob, (inp_w, inp_h))
        blob = blob.transpose(2, 0, 1)[None]
        return blob, scale, shift

    def preprocess(
        self,
        imgs: List[np.ndarray],
        boxes: List[cb.Box],
    ):
        if len(imgs) != len(boxes):
            raise ValueError(f"imgs and boxes should have same length, but got {len(imgs)} and {len(boxes)}")

        blobs = []
        scales = []
        shifts = []
        for img, box in zip(imgs, boxes):
            blob, scale, shift = self._transform(img, box)
            blobs.append(blob)
            scales.append(scale)
            shifts.append(shift)
        return blobs, scales, shifts

    def _reconstruct_vertices(
        self,
        param: np.ndarray,
        dense_flag: bool = False,
    ) -> np.ndarray:
        def _similar_transform(pts3d, scales):
            x_s, y_s = scales[..., 1], scales[..., 0]
            z_s = (x_s + y_s) / 2
            pts3d[..., 0] -= 1  # for Python compatibility
            pts3d[..., 2] -= 1
            pts3d[..., 1] = 120 - pts3d[..., 1]
            pts3d[..., 2] = pts3d[..., 2] + np.min(pts3d[..., 2])

            pts3d[..., 0] *= x_s
            pts3d[..., 1] *= y_s
            pts3d[..., 2] *= z_s
            return pts3d

        P, alpha_shp, alpha_exp, scales, shifts = [x for x in _parse_tffda_param(param)]
        if dense_flag:
            pts3d = self.bfm_engine(
                R=P[..., :3],
                offset=P[..., 3:],
                alpha_shp=alpha_shp,
                alpha_exp=alpha_exp,
            )["output"]
        else:
            param = self._u + self._w_shp @ alpha_shp + self._w_exp @ alpha_exp
            pts3d = P[..., :3] @ param.reshape(3, -1, order="F")
            pts3d += P[..., 3:]

        pts3d = pts3d.transpose(1, 0)
        pts3d = _similar_transform(pts3d, scales)
        pts3d[..., :2] += shifts
        return pts3d

    def _gen_3d_landmarks(self, params) -> List[np.ndarray]:
        lmks = np.stack([self._reconstruct_vertices(param, dense_flag=False) for param in params])
        return lmks

    def _gen_depth_imgs(self, imgs, params, tri) -> List[np.ndarray]:
        dense_vertices_list = [self._reconstruct_vertices(param, dense_flag=True) for param in params]
        depth_imgs = [depth(img, dense_vertices, tri) for img, dense_vertices in zip(imgs, dense_vertices_list)]
        return depth_imgs

    @staticmethod
    def _get_pose_degrees(params) -> List[np.ndarray]:
        pose_degrees = np.stack([calc_pose(param)[1] for param in params])
        return pose_degrees

    @staticmethod
    def _get_pose(pose_degrees) -> List[np.ndarray]:
        yaw, pitch, _ = pose_degrees

        pose = ""
        if yaw >= 20:
            pose += "Left"
        elif yaw <= -20:
            pose += "Right"
        elif pitch <= -20:
            pose += "Up"
        elif pitch >= 20:
            pose += "Down"

        if abs(yaw) <= 45:
            pose += "Frontal"
        else:
            pose += "Profile"

        return FacePose.obj_to_enum(pose).value

    def __call__(
        self,
        imgs: List[np.ndarray],
        boxes: List[cb.Box],
        return_depth: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        blobs, scales, shifts = self.preprocess(imgs, boxes)
        preds = {k: [] for k in self.engine.output_infos.keys()}
        for batch in cb.make_batch(blobs, self.batch_size):
            inputs = {
                name: np.concatenate(
                    append_to_batch(batch, self.batch_size),
                    dtype=v["dtype"],
                )
                for name, v in self.engine.input_infos.items()
            }
            tmp_preds = self.engine(**inputs)
            for k, v in tmp_preds.items():
                preds[k].append(detach_from_batch(v, len(batch)))
        preds = {k: np.concatenate(v, 0) for k, v in preds.items()}
        params = np.concatenate((preds["params"], scales, shifts), -1)
        lmk3d68pts = self._gen_3d_landmarks(params)
        pose_degrees = self._get_pose_degrees(params)

        if return_depth:
            # if self.depth_img_mode == "black":
            bgs = [np.zeros_like(img) for img in imgs]
            # else:
            #     bgs = imgs
            depth_imgs = self._gen_depth_imgs(bgs, params, self._tri)
        else:
            depth_imgs = [None] * len(imgs)

        poses = [self._get_pose(x) for x in pose_degrees]

        return [
            {
                "param": params[i],
                "lmk3d68pt": lmk3d68pts[i],
                "depth_img": depth_imgs[i],
                "pose_degree": pose_degrees[i],
                "pose": poses[i],
                "info": {
                    "model_fpath": self.model_path,
                },
            }
            for i in range(len(imgs))
        ]

    @staticmethod
    def draw_results(
        img: np.ndarray,
        boxes: List[cb.Box],
        results: Dict[str, np.ndarray],
        plot_details: bool = False,
    ) -> np.ndarray:
        if len(boxes) != len(results):
            raise ValueError(f"boxes and results should have same length, but got {len(boxes)} and {len(results)}")

        max_text_height = img.shape[0] // 30 + 1
        for result, face_box in zip(results, boxes):
            lmk2d68pt = cb.Keypoints(result["lmk3d68pt"][..., :2])
            depth_img = result["depth_img"]
            img = cb.draw_keypoints(img, lmk2d68pt)
            if depth_img is not None:
                img = cv2.addWeighted(img, 0.5, depth_img, 0.5, 0)

            if plot_details:
                text = (
                    f"yaw: {result['pose_degree'][0]:.2f}, "
                    f"pitch: {result['pose_degree'][1]:.2f}, "
                    f"roll: {result['pose_degree'][2]:.2f}"
                )
                text_height = np.clip(round(face_box.height * 0.05) + 1, 1, max_text_height)
                img = cb.draw_text(
                    img=img,
                    text=text,
                    location=face_box.left_bottom,
                    text_size=text_height,
                )

                text = f"pose: {FacePose(result['pose'].item()).name}"
                text_height = np.clip(round(face_box.height * 0.05) + 1, 1, max_text_height)
                img = cb.draw_text(
                    img=img,
                    text=text,
                    location=face_box.left_bottom + (0, text_height),
                    text_size=text_height,
                )

        if plot_details:
            detail = {
                "Model Path": results[0]["info"]["model_fpath"],
            }
            text = ""
            for k, v in detail.items():
                text += f"{k}: {v}\n"
            text_height = max(img.shape) // 100 + 1
            location = (5, 5)
            img = cb.draw_text(
                img=img,
                text=text,
                location=location,
                text_size=text_height,
            )

        return img
