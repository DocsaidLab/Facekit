from typing import Optional

import capybara as cb
import numpy as np
from matplotlib import colormaps

RAINBOW = colormaps["rainbow"]

ColorBar = (
    np.stack([np.array(RAINBOW(x / 600)) for x in range(600)])[:, :3] * 255
)  # red to blue (0 ~ 1)
ColorBar = ColorBar.round().astype("uint8")[:, None, :]
ColorBar = np.flip(ColorBar, axis=1)
ColorBar = np.tile(ColorBar, (1, 10, 1))
ColorBar = cb.pad(
    ColorBar,
    (20, 20, 5, 42),
    (0, 0, 0),
)

for x in np.linspace(620, 20, 11):
    score = (x - 20) / 600
    y = int(640 - x)
    ColorBar = cb.draw_text(
        ColorBar,
        f"- {score:.1f}",
        location=(15, y - 7),
        color=(255, 255, 255),
        text_size=14,
    )


def draw_results(
    image: np.ndarray,
    boxes: cb.Boxes,
    scores: Optional[np.ndarray] = None,
    kpts_list: Optional[cb.KeypointsList] = None,
    draw_scale: Optional[float] = 1.0,
    show_score: bool = False,
    show_score_bar: bool = True,
):
    plotted = image.copy()
    plotted = (
        (plotted - plotted.min()) / (plotted.max() - plotted.min() + 1e-8) * 255
    ).astype("uint8")
    img_h = image.shape[0]

    if len(boxes):
        scores = [None] * len(boxes) if scores is None else scores
        kpts_list = [None] * len(boxes) if kpts_list is None else kpts_list

        if len(boxes) != len(scores):
            raise ValueError("boxes and scores must have the same length")

        for box, score, kpts in zip(boxes, scores, kpts_list):
            color = (0, 255, 0)
            if score is not None:
                color = np.array(RAINBOW(1 - score)).flatten()[:3] * 255
                color = color.round().astype("uint8").tolist()
                text_size = np.clip(round(0.2 * box.height), 7, 24) * draw_scale
                stroke_width = np.clip(text_size // 8, 1, None)
                if show_score:
                    plotted = cb.draw_text(
                        plotted,
                        f"{score.item():.3f}",
                        location=box.left_top - [0, 1.05 * text_size],
                        color=(255, 255, 255),
                        text_size=text_size,
                        stroke_width=stroke_width,
                        stroke_fill=(0, 0, 0),
                    )

            thickness = np.clip(round(0.01 * box.height) + 1, 1, 3) * draw_scale
            plotted = cb.draw_box(plotted, box, color=tuple(color), thickness=thickness)

            if kpts is not None:
                scale = np.clip(box.height / img_h, 0, 2) * draw_scale
                plotted = cb.draw_keypoints(plotted, kpts, scale)

    img_h = plotted.shape[0]
    if show_score_bar:
        bar = cb.imresize(ColorBar.copy(), (img_h, None))
        plotted = np.concatenate((plotted, bar), axis=1)

    return plotted
