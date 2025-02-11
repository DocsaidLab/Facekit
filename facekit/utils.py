from typing import List

import capybara as cb
import numpy as np


def download_model_and_return_model_fpath(
    file_id: str,
    model_fname: str,
) -> cb.Path:
    target_folder = cb.Path.home() / ".cache/facekit"
    target_folder.mkdir(parents=True, exist_ok=True)
    model_fpath = target_folder / model_fname
    if not model_fpath.exists():
        cb.download_from_google(
            file_id=file_id,
            file_name=model_fname,
            target=target_folder,
        )
    return model_fpath


def append_to_batch(xs: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    remaid = len(xs) % batch_size
    if remaid:
        dummy_img = np.zeros_like(xs[0])
        xs.extend([dummy_img] * (batch_size - remaid))
    return xs


def detach_from_batch(xs: List[np.ndarray], length: int) -> List[np.ndarray]:
    return xs[:length]
