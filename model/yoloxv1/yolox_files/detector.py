import logging

import cv2
import numpy as np
import torch

from .yolo_pafpn import YOLOPAFPN
from .yolox_head import YOLOXHead
from .yolox import YOLOX


class Detector:
    def __init__(self, config):
        print(__name__)
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config["root"]
        self.yolox = self._create_yolox_model()

    def predict_object_bbox_from_image(self, class_names, image, detect_ids):
        image = self._preprocess_image(image)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()
        if self.config["use_gpu"]:
            image = image.cuda()
            if self.config["fp16"]:
                image = image.half()
        with torch.no_grad():
            outputs = self.yolox(image)

    def _create_yolox_model(self):
        self.input_res = (self.config["input_size"], self.config["input_size"])
        model_type = self.config["model_type"]
        model_path = (self.root_dir / self.config["model_files"][model_type]).resolve()
        model_sizes = self.config["model_sizes"][model_type]

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
            self.input_res,
            self.config["detect_ids"],
            self.config["iou_threshold"],
            self.config["score_threshold"],
        )
        return self._load_yolox_weights(model_path, model_sizes)

    def _get_model(self, model_sizes):
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(
            model_sizes["depth"], model_sizes["width"], in_channels=in_channels
        )
        head = YOLOXHead(
            self.config["num_classes"], model_sizes["width"], in_channels=in_channels
        )
        model = YOLOX(backbone, head)
        return model

    def _load_yolox_weights(self, model_path, model_settings):
        if model_path.is_file():
            ckpt = torch.load(str(model_path), map_location="cpu")
            model = self._get_model(model_settings)
            if self.config["use_gpu"]:
                model.cuda()
                if self.config["fp16"]:
                    model.half()
            model.eval()
            model.load_state_dict(ckpt["model"])
            return model

        raise ValueError(
            "Model file does not exist. Please check that %s exists." % model_path
        )

    def _preprocess_image(self, image, swap=(2, 0, 1)):
        print(self.input_res)
        if len(image.shape) == 3:
            padded_img = (
                np.ones((self.input_res[0], self.input_res[1], 3), dtype=np.uint8)
                * 114
            )
        else:
            padded_img = np.ones(self.input_res, dtype=np.uint8) * 114

        r = min(
            self.input_res[0] / image.shape[0], self.input_res[1] / image.shape[1]
        )
        resized_img = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img
