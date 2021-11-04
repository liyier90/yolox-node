import logging

import cv2
import numpy as np
import torch
import torchvision

from .yolo_pafpn import YOLOPAFPN
from .yolox_head import YOLOXHead
from .yolox import YOLOX

NUM_CHANNELS = 3


class Detector:
    def __init__(self, config):
        print(__name__)
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.root_dir = config["root"]
        self.yolox = self._create_yolox_model()

    def predict_object_bbox_from_image(self, image, class_names, detect_ids):
        image, scale = self._preprocess_image(image)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()
        if self.config["use_gpu"]:
            image = image.cuda()
            if self.config["fp16"]:
                image = image.half()
        with torch.no_grad():
            predictions = self.yolox(image)
            predictions = self._postprocess_predictions(predictions, scale)

        return predictions, scale

    def _create_yolox_model(self):
        self.input_size = (self.config["input_size"], self.config["input_size"])
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
            self.input_size,
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

    def _postprocess_predictions(self, predictions, scale):
        # xywh to xyxy
        bboxes = torch.empty_like(predictions)
        bboxes[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
        bboxes[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
        bboxes[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
        bboxes[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
        predictions[:, :, :4] = bboxes[:, :, :4]

        outputs = [None for _ in range(len(predictions))]
        for i, prediction in enumerate(predictions):
            # If none are remaining => process next image
            if not prediction.size(0):
                continue
            # Get score and class with highest confidence
            class_score, class_pred = torch.max(
                prediction[:, 5 : 5 + self.config["num_classes"]], 1, keepdim=True
            )

            conf_mask = (
                prediction[:, 4] * class_score.squeeze()
                >= self.config["score_threshold"]
            ).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (prediction[:, :5], class_score, class_pred.float()), 1
            )
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            # Class agnostic NMS
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                self.config["iou_threshold"],
            )

            detections = detections[nms_out_index]
            if outputs[i] is None:
                outputs[i] = detections
            else:
                outputs[i] = torch.cat((outputs[i], detections))

        return outputs

    def _preprocess_image(self, image):
        # Initialize canvas for padded image as gray
        padded_img = np.full(
            (self.input_size[0], self.input_size[1], NUM_CHANNELS), 114, dtype=np.uint8
        )
        scale = min(
            self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1]
        )
        scaled_height = int(image.shape[0] * scale)
        scaled_width = int(image.shape[1] * scale)
        resized_img = cv2.resize(
            image,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[:scaled_height, :scaled_width] = resized_img

        # Rearrange from (H, W, C) to (C, H, W)
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, scale
