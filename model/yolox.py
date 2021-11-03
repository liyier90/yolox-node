from .yoloxv1 import yolox_model


class Node:
    def __init__(self, config, **kwargs):
        print(__name__)
        self.config = config
        self.model = yolox_model.YOLOXModel(self.config)

    def run(self, inputs):
        ret = self.model.predict(inputs["img"])
        return ret