from .yoloxv1 import yolox_model


class Node:
    def __init__(self, config, **kwargs):
        print(__name__)
        self.config = config
        self.model = yolox_model.YOLOXModel(self.config)

    def run(self, inputs):
        bboxes, labels, scores = self.model.predict(inputs["img"])
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}

        return outputs
