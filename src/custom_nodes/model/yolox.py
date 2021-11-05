# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High performance anchor-free YOLO object detection model"""

from pathlib import Path
from typing import Any, Dict

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
        input_size (:obj:`int`): **Default=416.** Input image resolution of the
            YOLOX model.
        detect_ids (:obj:`List[int]`): **Default=[0].** List of object category
            IDs to be detected.
        iou_threshold (:obj:`float`): **[0, 1], default = 0.45.** Overlapping
            bounding boxes with Intersection over Union (IoU) above the
            threshold will be discarded.
        score_threshold (:obj:`float`): **[0, 1], default = 0.25.** Bounding
            boxes with confidence score (product of objectness score and
            classification score) below the threshold will be discarded.
        fp16 (:obj:`bool`): **Default = False.** Flag to determine if
            half-precision floating-point should be used for inference
        fuse (:ojb:`bool`): **Default = False.** Flag to determine if the
            convolution and batch normalization layers should be fused for
            inference.

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
