import capybara as cb
from fire import Fire

import facekit as fk

cur_folder = cb.get_curdir(__file__)


def main(
    src_path: str = str(cur_folder / "data" / "EmmaWatson1.jpg"),
    tgt_path: str = str(cur_folder / "data" / "face_bank" / "EmmaWatson.jpg"),
):
    face_detection = fk.build_face_detection()
    face_recognition = fk.build_face_recognition()
    face_compare = fk.FaceCompare(face_recognition.mapping_table)
    src_img = cb.imread(src_path)
    tgt_img = cb.imread(tgt_path)
    lmks = [p["lmk5pts"][0] for p in face_detection([src_img, tgt_img])]

    out_folder = cur_folder / "output"
    out_folder.mkdir(exist_ok=True, parents=True)
    results = face_recognition([src_img, tgt_img], lmks)
    src_emb = results[0]["embeddings"]
    tgt_emb = results[1]["embeddings"]
    sim, is_same = face_compare(src_emb, tgt_emb)
    print(f"Similarity: {sim:.5f}, Is same: {is_same}")


if __name__ == "__main__":
    Fire(main)
