import numpy as np

from .yolox_files.detector import Detector


class YOLOXModel:
    def __init__(self, config):
        print(__name__)

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
    def detect_ids(self):
        return self._detect_ids

    @detect_ids.setter
    def detect_ids(self, ids):
        if not isinstance(ids, list):
            raise TypeError("detect_ids has to be a list")
        if not ids:
            print("Detecting all available classes")
        self._detect_ids = ids

    def predict(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_object_bbox_from_image(image, self.class_names)
