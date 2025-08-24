from typing import Dict, List, Tuple, Union

import capybara as cb
import numpy as np
from huggingface_hub import hf_hub_download


def download_model_and_return_model_fpath(
    repo_id: str,
    model_fname: str,
) -> cb.Path:
    target_folder = cb.Path.home() / ".cache/pyface"
    target_folder.mkdir(parents=True, exist_ok=True)
    model_fpath = hf_hub_download(
        repo_id=repo_id,
        filename=model_fname,
        cache_dir=str(target_folder),
    )
    return cb.Path(model_fpath)


def append_to_batch(xs: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    remaid = len(xs) % batch_size
    if remaid:
        dummy_img = np.zeros_like(xs[0])
        xs.extend([dummy_img] * (batch_size - remaid))
    return xs


def detach_from_batch(xs: List[np.ndarray], length: int) -> List[np.ndarray]:
    return xs[:length]


def get_mapped_score(
    score: Union[float, np.number, List[float], np.ndarray],
    mapping_table: List[Tuple[float, float]],
):
    if mapping_table[0][1] != 1.0:
        mapping_table = [[1, 1]] + mapping_table

    if mapping_table[-1][1] != 0.0:
        mapping_table = mapping_table + [[0, 0]]

    src_values, dst_values = np.array(mapping_table).T
    for idx, src_value in enumerate(src_values):
        if score > src_value:
            break

    factor = (score - src_values[idx]) / (src_values[idx - 1] - src_values[idx])
    return dst_values[idx] + factor * (dst_values[idx - 1] - dst_values[idx])


def get_mapped_scores(
    scores: Union[List[float], np.ndarray],
    mapping_tables: Dict[str, List[Tuple[float, float]]],
) -> Union[List[float], np.ndarray]:
    is_np = isinstance(scores, np.ndarray)
    if not is_np:
        scores = np.array(scores)
    if scores.ndim != 1:
        raise ValueError(f"Expected scores to be 1D, but got {scores.ndim}D")

    mscores = []
    for i, score in enumerate(scores):
        if i in mapping_tables:
            mscore = get_mapped_score(score, mapping_tables[i])
        else:
            mscore = score
        mscores.append(mscore)
    mscores = np.array(mscores)
    if not is_np:
        mscores = mscores.tolist()
    return mscores
