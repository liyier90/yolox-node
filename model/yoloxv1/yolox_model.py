from .yolox_files.detector import Detector


class YOLOXModel:
    def __init__(self, config):
        print(__name__)

        with open((config["root"] / config["classes"]).resolve()) as infile:
            self.class_names = [line.strip() for line in infile.readlines()]
        self.detect_ids = config["detect_ids"]

        self.detector = Detector(config)

    def predict(self, image):
        return self.detector.predict_object_bbox_from_image(
            self.class_names, image, self.detect_ids
        )
