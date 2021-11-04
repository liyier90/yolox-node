"""Evaluate performance of YOLOX"""

import logging
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# import torch
import yaml

# from loguru import logger

# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp.yolox_base import Exp
# from yolox.models.yolox import YOLOX
# from yolox.utils import (
#     # fuse_model,
#     get_model_info,
#     postprocess,
#     vis,
# )

from model.yolox import Node as YOLOXNode

# DETECT_IDS = (0,)
# IMAGE_EXTS = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# NUM_FRAMES = 10

# fmt: off
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
# fmt: on


def visualize(output, raw_img, ratio, class_names):
    img = raw_img
    if output is None:
        return img
    output = output.cpu()
    bboxes = output[:, :4]
    # preprocessing: resize
    bboxes /= ratio
    classes = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    for i in range(len(bboxes)):
        box = bboxes[i]
        cls_id = int(classes[i])
        score = scores[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1,
        )
        cv2.putText(
            img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1
        )

    return img


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    PROJ_DIR = Path(__file__).resolve().parent

    with open(PROJ_DIR / "eval_config.yml") as infile:
        CONFIG = yaml.safe_load(infile)
    CONFIG["root"] = PROJ_DIR
    # External configs
    CONFIG["input_dir"] = Path(
        "~/code/YOLOX/data/video/multiple_people.mp4"
    ).expanduser()
    CONFIG["output_dir"] = PROJ_DIR / "YOLOX_outputs"
    # print(CONFIG)

    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    # input.recorded
    cap = cv2.VideoCapture(str(CONFIG["input_dir"]))
    _, frame = cap.read()
    cv2.imwrite(str(CONFIG["output_dir"] / "before_frame.jpg"), frame)
    inputs = {"img": frame}

    yolox_node = YOLOXNode(CONFIG)
    outputs, ratio = yolox_node.run(inputs)

    result_frame = visualize(outputs[0], frame.copy(), ratio, yolox_node.model.class_names)
    cv2.imwrite(str(CONFIG["output_dir"] / "after_frame.jpg"), result_frame)

    # main(EXP_OBJ, CONFIG)
