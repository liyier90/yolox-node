"""High performance anchor-free YOLO object detection model"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode

from .yoloxv1 import yolox_model


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Node class to initialize and use YOLOX to infer from an image frame.

    The YOLOX node is capable detecting objects from 80 categories. The table
    of object categories can be found :term:`here <object detection indices>`.
    The "yolox-tiny" model is used by default and can be changed to one of
    (yolox-tiny/yolox-s/yolox-m/yolox-l).

    Inputs:
        |img|

    Outputs:
        |bboxes|

        |bbox_labels|

        |bbox_scores|

    Configs:
        model_type (:obj:`str`): **{"yolox-tiny", "yolox-s", "yolox-m",
            "yolox-l"}, default="yolox-tiny".** Defines the type of YOLOX model
            to be used.
        TODO: TBD

    References:
        YOLOX: Exceeding YOLO Series in 2021:
            https://arxiv.org/abs/2107.08430
        Inference code and model weights:
            https://github.com/Megvii-BaseDetection/YOLOX
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.config = config
        self.config["root"] = Path(__file__).resolve().parents[3]
        self.model = yolox_model.YOLOXModel(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Read `img` from `inputs` and return the bboxes of the detect objects.

        The classes of objects to be detected can be specified through the
        `detect_ids` configuration option.

        Args:
            inputs (Dict): Inputs dictionary with the key `img`.

        Returns:
            (Dict): Outputs dictionary with the keys `bboxes`, `bbox_labels`,
                and `bbox_scores`.
        """
        bboxes, labels, scores = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return outputs
