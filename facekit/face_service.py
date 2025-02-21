from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import capybara as cb
import numpy as np

from .components import (
    FaceCompare,
    FaceNormalize,
    RecogLevel,
    build_face_depth,
    build_face_detection,
    build_face_landmark,
    build_face_recognition,
)
from .object import TDDFA, Encode, Face, Faces, Who

__all__ = ["FaceService"]


class FaceService:
    def __init__(
        self,
        batch_size: int = 1,
        enable_recognition: bool = False,
        enable_depth: bool = False,
        enable_landmark: bool = False,
        detect_kwargs: Optional[Dict] = {},
        depth_kwargs: Optional[Dict] = {},
        recognition_kwargs: Optional[Dict] = {},
        landmark_kwargs: Optional[Dict] = {},
        face_bank: Optional[Union[str, Path]] = None,
        recog_level: Union[RecogLevel, str] = RecogLevel.High,
        # enable_fas: bool = False,
        # fas_kwargs: Optional[Dict] = {},
    ):
        self.detector = build_face_detection(batch_size=batch_size, **detect_kwargs)
        self.landmarker = build_face_landmark(**landmark_kwargs) if enable_landmark else None
        self.depther = build_face_depth(batch_size=batch_size, **depth_kwargs) if enable_depth else None

        if enable_recognition:
            self.norm = FaceNormalize()
            self.recognizer = build_face_recognition(batch_size=batch_size, **recognition_kwargs)
            self.comparer = FaceCompare(
                mapping_table=self.recognizer.mapping_table,
                method=self.recognizer.compare_method,
                recog_level=recog_level,
            )
            if face_bank is not None:
                self.face_bank = self._gen_face_bank(face_bank)

        # if enable_fas:
        #     if fas_model is not None:
        #         self.fas = fas_model
        #     else:
        #         self.fas = build_fas(**fas_kwargs, **engine_kwargs)

    @staticmethod
    def _flatten_imgs_and_proposals_list(
        imgs, proposals_list: List[Dict[str, np.ndarray]]
    ) -> Tuple[List[np.ndarray], cb.Boxes, cb.KeypointsList, List[int]]:
        out_imgs = []
        out_boxes = []
        out_lmk5pts = []
        for _, (img, proposals) in enumerate(zip(imgs, proposals_list)):
            if len(proposals["boxes"]):
                out_imgs.extend([img] * len(proposals["boxes"]))
                out_boxes.extend(proposals["boxes"])
                out_lmk5pts.extend(proposals["lmk5pts"])
        out_boxes = cb.Boxes(out_boxes)
        out_lmk5pts = cb.KeypointsList(out_lmk5pts)
        return out_imgs, out_boxes, out_lmk5pts

    def _fill_results_to_faces_list(
        self,
        faces_list: List[Faces],
        proposals_list: List[Dict[str, np.ndarray]],
        lmk_results: Optional[List[Dict[str, np.ndarray]]] = None,
        dep_results: Optional[List[Dict[str, np.ndarray]]] = None,
        enc_results: Optional[List[Dict[str, np.ndarray]]] = None,
    ) -> List[Faces]:
        i = 0
        for faces, proposals in zip(faces_list, proposals_list):
            faces.faces = [
                Face(box=cb.Box(box), lmk5pt=cb.Keypoints(lmk), score=score)
                for box, lmk, score in zip(proposals["boxes"], proposals["lmk5pts"], proposals["scores"])
            ]
            for face in faces:
                if lmk_results is not None:
                    lmk_result = lmk_results[i]
                    face.lmk68pt = cb.Keypoints(lmk_result["lmk"])
                if dep_results is not None:
                    dep_result = dep_results[i]
                    face.tddfa = TDDFA(
                        param=dep_result["param"],
                        lmk68pt=dep_result["lmk3d68pt"],
                        depth_img=dep_result["depth_img"],
                        pose=dep_result["pose"],
                        yaw=dep_result["pose_degree"][0],
                        roll=dep_result["pose_degree"][1],
                        pitch=dep_result["pose_degree"][2],
                    )
                if enc_results is not None:
                    enc_result = enc_results[i]
                    face.encoding = Encode(
                        vector=enc_result["embeddings"],
                        version=enc_result["info"]["version"],
                    )
                    face.norm_img = enc_result["norm_img"]
                # if fas_results is not None:
                #     fas_results = fas_results[i]
                #     face.liveness = Liveness(
                #         is_true=fas_results["pred_label"] == "Live",
                #         value=fas_results["score"],
                #         threshold=fas_results["info"]["threshold"],
                #         label=fas_results["pred_label"],
                #     )
                i += 1

        return faces_list

    def __call__(self, imgs: List[np.ndarray], do_1n: bool = False) -> List[Faces]:
        faces_list = [None] * len(imgs)

        if len(imgs):
            faces_list = [Faces(raw_image=img) for img in imgs]

            proposals_list = self.detector(imgs=imgs)
            imgs, boxes, lmk5pts = self._flatten_imgs_and_proposals_list(imgs, proposals_list)

            lmk_results = None
            depth_results = None
            enc_results = None

            if len(boxes):
                if self.landmarker is not None:
                    lmk_results = self.landmarker(imgs=imgs, boxes=boxes)

                if self.depther is not None:
                    depth_results = self.depther(imgs=imgs, boxes=boxes, return_depth=True)

                if self.recognizer is not None:
                    enc_results = self.recognizer(imgs=imgs, lmk5pts=lmk5pts)

                # if self.fas is not None:
                #     fas_results = self.fas(imgs=imgs, lmk5pts=lmk5pts, boxes=boxes)

            faces_list = self._fill_results_to_faces_list(
                faces_list=faces_list,
                proposals_list=proposals_list,
                lmk_results=lmk_results,
                dep_results=depth_results,
                enc_results=enc_results,
                # fas_results,
            )
            if do_1n and self.recognizer is not None:
                faces_list = self.do_1n(faces_list)

        return faces_list

    def do_1n(
        self,
        faces_list: List[Faces],
    ) -> List[Faces]:
        if self.face_bank is not None and len(self.face_bank):
            for faces in faces_list:
                for face in faces:
                    sorted_results = sorted(
                        [
                            (*self.comparer(face.encoding.vector, candidate.encoding.vector), ind)
                            for ind, candidate in enumerate(self.face_bank)
                        ],
                        reverse=True,
                    )
                    best_matched = sorted_results[0]
                    if best_matched[1]:
                        ind = best_matched[2]
                        face.who = Who(
                            name=self.face_bank[ind].who.name,
                            confidence=best_matched[0],
                            recognized_level=self.comparer.recognition_level,
                        )
        else:
            raise ValueError("self.face_bank is None, which is unsupport to do 1:N.")
        return faces_list

    def _gen_face_bank(self, bank_folder: Union[str, cb.Path]):
        face_bank = []
        files = cb.get_files(bank_folder, suffix=[".jpg", "jpeg", ".png"], return_pathlib=True)
        names = [f.stem for f in files]
        imgs = [cb.imread(file) for file in files]
        faces_list = self(imgs)
        for name, faces in zip(names, faces_list):
            if len(faces):
                face = faces[0]
                face.who = Who(
                    name=name,
                    confidence=1.0,
                    recognized_level=self.comparer.recognition_level,
                )
                face_bank.append(face)
        return face_bank

    def update_face_bank(self, face_bank: Union[str, Path]):
        if self.face_bank is None:
            self._gen_face_bank(face_bank)
        else:
            self.face_bank.extend(self._gen_face_bank(face_bank))

    def save_face_bank(self, out_json):
        cb.dump_json([x.be_jsonable() for x in self.face_bank], out_json)
