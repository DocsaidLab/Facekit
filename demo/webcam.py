from functools import partial

import capybara as cb
from fire import Fire

import pyface as pf

cur_dir = cb.get_curdir(__file__)


def process_frame(frame, face_service: pf.FaceService):
    timer = cb.Timer()
    timer.tic()
    faces = face_service([frame])[0]
    delta = timer.toc()
    frame = faces.gen_info_img()
    frame = cb.draw_text(frame, f"FPS: {1 / delta:.0f}", (0, 0), color=(255, 255, 255), text_size=24)
    return frame


def main(camera_ip: str = 0):
    kwargs = {"backend": "cuda"}
    face_service = pf.FaceService(
        enable_depth=True,
        enable_landmark=True,
        enable_recognition=True,
        enable_gender=True,
        detect_kwargs=kwargs,
        landmark_kwargs=kwargs,
        depth_kwargs=kwargs,
        recognition_kwargs=kwargs,
        face_bank=cur_dir / "data" / "face_bank",
    )
    demo = cb.WebDemo(camera_ip=camera_ip, pipelines=[partial(process_frame, face_service=face_service)])
    demo.run()


if __name__ == "__main__":
    Fire(main)
