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
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Half precision only supported on CUDA
        self.half = self.config["fp16"] and self.device.type != "cpu"
        self.yolox = self._create_yolox_model()

    def predict_object_bbox_from_image(self, image, class_names):
        image, scale = self._preprocess(image)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float().to(self.device)
        if self.config["fp16"]:
            image = image.half()
        with torch.no_grad():
            prediction = self.yolox(image)[0]
            bboxes, classes, scores = self._postprocess(
                prediction, scale, class_names
            )

        return bboxes, classes, scores

    def _create_yolox_model(self):
        self.detect_ids = torch.Tensor(self.config["detect_ids"])
        if self.config["use_gpu"]:
            self.detect_ids = self.detect_ids.cuda()
            if self.config["fp16"]:
                self.detect_ids = self.detect_ids.half()
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
            model = model.to(self.device)
            if self.config["fp16"]:
                model.half()
            model.eval()
            model.load_state_dict(ckpt["model"])
            return model

        raise ValueError(
            "Model file does not exist. Please check that %s exists." % model_path
        )

    def _postprocess(self, prediction, scale, class_names):
        # xywh to xyxy
        box_corners = torch.empty_like(prediction)
        box_corners[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
        box_corners[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
        box_corners[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
        box_corners[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
        prediction[:, :4] = box_corners[:, :4]

        # Get score and class with highest confidence
        class_score, class_pred = torch.max(
            prediction[:, 5 : 5 + self.config["num_classes"]], 1, keepdim=True
        )
        # Filter by score_threshold
        conf_mask = (
            prediction[:, 4] * class_score.squeeze() >= self.config["score_threshold"]
        ).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((prediction[:, :5], class_score, class_pred.float()), 1)
        detections = detections[conf_mask]
        # Early return if all are below score_threshold
        if not detections.size(0):
            return np.empty(0), np.empty(0), np.empty(0)

        # Class agnostic NMS
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            self.config["iou_threshold"],
        )
        output = detections[nms_out_index]

        # Filter by detect ids
        output = (
            output[torch.isin(output[:, 6], self.detect_ids)].cpu().detach().numpy()
        )
        bboxes = output[:, :4] / scale
        scores = output[:, 4] * output[:, 5]
        classes = np.array([class_names[int(i)] for i in output[:, 6]])

        return bboxes, classes, scores

    def _preprocess(self, image):
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
