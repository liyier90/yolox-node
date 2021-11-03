import logging

import torch

from .yolo_pafpn import YOLOPAFPN
from .yolox_head import YOLOXHead
from .yolox import YOLOX


SETTINGS = {
    "yolox-tiny": {
        "depth": 0.33,
        "width": 0.375,
    },
    "yolox-l": {
        "depth": 1.0,
        "width": 1.0,
    },
}


class Detector:
    def __init__(self, config):
        print(__name__)
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config["root"]
        self.yolox = self._create_yolox_model()

    def _create_yolox_model(self):
        model_type = self.config["model_type"]
        model_path = (self.root_dir / self.config["model_files"][model_type]).resolve()
        model_settings = SETTINGS[model_type]

        # TODO Log more configs
        self.logger.info(
            (
                "YOLOX model loaded with the following configs:\n\t"
                "Model type: %s\n\t"
                "Input resolution: %s\n\t"
                "IDs being detected: %s\n\t"
                "IOU threshold: %s\n\t"
                "Score threshold: %s\n\t"
            ),
            self.config["model_type"],
            self.config["input_size"],
            self.config["detect_ids"],
            self.config["iou_threshold"],
            self.config["score_threshold"],
        )
        return self._load_yolox_weights(model_path, model_settings)

    def _load_yolox_weights(self, model_path, model_settings):
        if model_path.is_file():
            ckpt = torch.load(str(model_path), map_location="cpu")
            # if self.config["device"] == "gpu":
            model = self.get_model(model_settings)
            model.load_state_dict(ckpt["model"])
            return model
        raise ValueError(
            "Model file does not exist. Please check that %s exists." % model_path
        )

    def get_model(self, model_settings):
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(
            model_settings["depth"], model_settings["width"], in_channels=in_channels
        )
        head = YOLOXHead(
            self.config["num_classes"], model_settings["width"], in_channels=in_channels
        )
        model = YOLOX(backbone, head)
        return model
