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

"""YOLOX models with model types: yolox-tiny, yolox-s, yolox-m, yolox-l"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .yolox_files.detector import Detector


class YOLOXModel:
    """Validates configuration, loads YOLOX model, and performs inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.

    Attributes:
        class_names (List[str]): Human-friendly class names of the object
            categories.
        detect_ids (List[int]): List of selected object category IDs. IDs not
            found in this list will be filtered away from the results. An empty
            list indicates that all object categories should be detected.
        detector (Detector): YOLOX detector object to infer bboxes from a
            provided image frame.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # Check threshold values
        if not 0 <= config["score_threshold"] <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0 <= config["iou_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")

        # Check for YOLOX weights

        with open((config["root"] / config["classes"]).resolve()) as infile:
            self.class_names = [line.strip() for line in infile.readlines()]
        self.detect_ids = config["detect_ids"]

        self.detector = Detector(config)

    @property
    def detect_ids(self) -> List[int]:
        """The list of selected object category IDs"""
        return self._detect_ids

    @detect_ids.setter
    def detect_ids(self, ids):
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        if not ids:
            print("Detecting all available classes")
        self._detect_ids = ids

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Predicts bboxes from image.

        Args:
            image (np.array): Input image frame

        Returns:
            (Tuple[List[np.array], List[str], List[float]]): Returned tuple
                contains:
                - A list of detection bboxes
                - A list of human-friendly detection class names
                - A list of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_object_bbox_from_image(image, self.class_names)
