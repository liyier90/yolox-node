from .yolox_files.detector import Detector

class YOLOXModel:
    def __init__(self, config):
        print(__name__)
        self.detector = Detector(config)