"""Evaluate performance of YOLOX"""

import logging
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

# import cv2
import numpy as np
# import torch
import yaml

# from loguru import logger

# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp.yolox_base import Exp
# from yolox.models.yolox import YOLOX
# from yolox.utils import (
#     # fuse_model,
#     get_model_info,
#     postprocess,
#     vis,
# )

from model.yolox import Node as YOLOXNode

# DETECT_IDS = (0,)
# IMAGE_EXTS = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# NUM_FRAMES = 10


class Exp:
    """Base class for YOLOX experiments."""

    def __init__(
        self,
        depth,
        width,
        # conf_thres,
        # nms_thres,
        # test_size,
    ):
        # ---------------- model config ---------------- #
        self.num_classes = 80
        self.depth = depth
        self.width = width

        # ---------------- dataloader config ---------------- #
        # # set worker to 4 for shorter dataloader init time
        # self.data_num_workers = 4
        # self.input_size = (640, 640)  # (height, width)
        # # Actual multiscale ranges: [640-5*32, 640+5*32].
        # # To disable multiscale training, set the
        # # self.multiscale_range to 0.
        # self.multiscale_range = 5
        # # You can uncomment this line to specify a multiscale range
        # # self.random_size = (14, 26)
        # self.data_dir = None
        # self.train_ann = "instances_train2017.json"
        # self.val_ann = "instances_val2017.json"

        # -----------------  testing config ------------------ #
        # self.conf_thres = conf_thres
        # self.nms_thres = nms_thres
        # self.test_size = (test_size, test_size)
        self.conf_thres = 0.01
        self.nms_thres = 0.65
        self.test_size = (640, 640)

    # def get_model(self) -> nn.Module:
    #     from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    #     def init_yolo(M):
    #         for m in M.modules():
    #             if isinstance(m, nn.BatchNorm2d):
    #                 m.eps = 1e-3
    #                 m.momentum = 0.03

    #     if getattr(self, "model", None) is None:
    #         in_channels = [256, 512, 1024]
    #         backbone = YOLOPAFPN(
    #             self.depth, self.width, in_channels=in_channels, act=self.act
    #         )
    #         head = YOLOXHead(
    #             self.num_classes, self.width, in_channels=in_channels, act=self.act
    #         )
    #         self.model = YOLOX(backbone, head)

    #     self.model.apply(init_yolo)
    #     self.model.head.initialize_biases(1e-2)
    #     return self.model


def get_exp(exp_name):
    model_config_dict = {
        "yolox-tiny": {"depth": 0.33, "width": 0.375},
        "yolox-l": {"depth": 1.0, "width": 1.0},
    }
    return Exp(**model_config_dict[exp_name])


# def filter_by_detect_id(classes: torch.Tensor) -> np.array:
#     """Return a mask tensor containing rows with the allowed class id"""
#     return np.isin(classes.cpu().detach().numpy(), DETECT_IDS)


# class FpsCounter:
#     """FPS node class that calculates the FPS of the image frame.

#     This node calculates instantaneous FPS and a 10 frame moving average
#     FPS. A preferred output setting can be set via the config file.
#     """

#     def __init__(self):
#         self.count = 0
#         self.fps_log_freq = 100
#         self.global_avg_fps = 0.0
#         self.prev_frame_timestamp = 0.0
#         self.moving_average_fps: List[float] = []

#     def run(self, pipeline_end: bool = False) -> None:
#         """Calculates FPS using the time difference between the current
#         frame and the previous frame."""
#         curr_frame_timestamp = time.perf_counter()
#         # Frame level FPS
#         frame_fps = 1.0 / (curr_frame_timestamp - self.prev_frame_timestamp)
#         self.prev_frame_timestamp = curr_frame_timestamp
#         # Calculate moving average FPS (dampen_fps)
#         average_fps = self._moving_average(frame_fps)
#         if pipeline_end:
#             logger.info(f"Avg FPS over all processed frames: {self.global_avg_fps:.2f}")
#         else:
#             if self.count > 0 and self.count % self.fps_log_freq == 0:
#                 logger.info(f"Average FPS over last 10 frames: {average_fps:.2f}")
#             self.global_avg_fps = self._global_average(frame_fps)

#     def _moving_average(self, frame_fps: float) -> float:
#         self.moving_average_fps.append(frame_fps)
#         if len(self.moving_average_fps) > NUM_FRAMES:
#             self.moving_average_fps.pop(0)
#         moving_average_val = mean(self.moving_average_fps)
#         return moving_average_val

#     def _global_average(self, frame_fps: float) -> float:
#         # Cumulative moving average formula
#         global_average = (frame_fps + self.count * self.global_avg_fps) / (
#             self.count + 1
#         )
#         self.count += 1
#         return global_average


# class Predictor:
#     """Object to perform inference using YOLOX models and post-process the
#     results
#     """

#     def __init__(
#         self,
#         model: YOLOX,
#         exp: Exp,
#         cls_names: Tuple = COCO_CLASSES,
#         trt_file: Path = None,
#         decoder: Callable = None,
#         device: str = "cpu",
#         fp16: bool = False,
#         legacy: bool = False,
#     ):
#         self.model = model

#         self.conf_thres = exp.test_conf
#         self.nms_thres = exp.nmsthre
#         self.num_classes = exp.num_classes
#         self.test_size = exp.test_size

#         self.cls_names = cls_names
#         self.decoder = decoder
#         self.device = device
#         self.fp16 = fp16
#         self.preprocess = ValTransform(legacy=legacy)
#         # if trt_file is not None:
#         #     from torch2trt import TRTModule
#         #     model_trt = TRTModule()
#         #     model_trt.load_state_dict(torch.load(trt_file))
#         #     x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
#         #     self.model(x)
#         #     self.model = model_trt
#         self.fps_counter = FpsCounter()

#     def infer(self, img: np.array) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
#         """Perform inference on one image. Post-processing of detection output
#         is done using iou > nms_thres and obj_score * cls_score > conf_thres.

#         Args:
#             img: Input image for inference

#         Returns:
#             (Tuple[List[torch.Tensor], Dict[str, Any]]):
#                 List[torch.Tensor]: Post-processed inference detection output.
#                     Each row of the tensor contains 7 elements:
#                     - 0, 1, 2, 3: bbox corners
#                     - 4: objectness score
#                     - 5: classification score
#                     - 6: class_id
#                 Dict[str, Any]: Information about the input image
#         """
#         img_info: Dict[str, Any] = {"id": 0}
#         if isinstance(img, Path):
#             img_info["file_name"] = img.name
#             img = cv2.imread(str(img))
#         else:
#             img_info["file_name"] = None
#         height, width = img.shape[:2]
#         img_info["height"] = height
#         img_info["width"] = width
#         img_info["raw_img"] = img

#         ratio = min(self.test_size[0] / height, self.test_size[1] / width)
#         img_info["ratio"] = ratio

#         img, _ = self.preprocess(img, None, self.test_size)
#         img = torch.from_numpy(img).unsqueeze(0)
#         img = img.float()
#         if self.device == "gpu":
#             img = img.cuda()
#             if self.fp16:
#                 img = img.half()

#         with torch.no_grad():
#             time_start = time.time()
#             outputs = self.model(img)
#             if self.decoder is not None:
#                 outputs = self.decoder(outputs, dtype=outputs.type())
#             outputs = postprocess(
#                 outputs,
#                 self.num_classes,
#                 self.conf_thres,
#                 self.nms_thres,
#                 class_agnostic=True,
#             )
#             outputs = [
#                 output[filter_by_detect_id(output[..., 6])] for output in outputs
#             ]
#             # logger.info(f"Infer time: {time.time() - time_start:.4f}")

#         self.fps_counter.run()

#         return outputs, img_info

#     def visualize(
#         self,
#         output: torch.Tensor,
#         img_info: Dict[str, Any],
#         cls_conf: Optional[float] = None,
#     ) -> np.array:
#         """Visualize detection result. A second confidence filtering can be
#         done here
#         """
#         if cls_conf is None:
#             cls_conf = self.conf_thres
#         img = img_info["raw_img"]
#         ratio = img_info["ratio"]
#         if output is None:
#             return img
#         output = output.cpu()
#         bboxes = output[:, :4]
#         # preprocessing: resize
#         bboxes /= ratio
#         classes = output[:, 6]
#         scores = output[:, 4] * output[:, 5]

#         return vis(img, bboxes, scores, classes, cls_conf, self.cls_names)


# def get_image_list(path: Path) -> List[Path]:
#     """Get a list of images from the given directory"""
#     image_names = []
#     for file_path in path.iterdir():
#         if file_path.suffix in IMAGE_EXTS:
#             image_names.append(file_path)
#     return image_names


# def evaluate_image(
#     predictor: Predictor, vis_dir: Path, path: Path, current_time: time.struct_time
# ) -> None:
#     """Run evaluation on a single image or a folder of images"""
#     save_dir = vis_dir / time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     if path.is_dir():
#         files = sorted(get_image_list(path))
#     else:
#         files = [path]
#     for image_path in files:
#         outputs, img_info = predictor.infer(image_path)
#         result_image = predictor.visualize(outputs[0], img_info)
#         save_file_name = save_dir / image_path.name
#         logger.info(f"Saving detection in {save_file_name}")
#         cv2.imwrite(str(save_file_name), result_image)


# def evaluate_imageflow(
#     predictor: Predictor,
#     vis_dir: Path,
#     current_time: time.struct_time,
#     config: SimpleNamespace,
# ):
#     """Run evaluation on a video/live feed"""
#     cap = cv2.VideoCapture(str(config.path) if config.mode == "video" else config.camid)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_size = (
#         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#     )
#     save_dir = vis_dir / time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     if config.mode == "video":
#         save_path = save_dir / config.path.name
#     else:
#         save_path = save_dir / "camera.mp4"
#     logger.info(f"Video save_path is {save_path}")
#     vid_writer = cv2.VideoWriter(
#         str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
#     )
#     while True:
#         ret_val, frame = cap.read()
#         if ret_val:
#             outputs, img_info = predictor.infer(frame)
#             result_frame = predictor.visualize(
#                 outputs[0], img_info, predictor.conf_thres
#             )
#             vid_writer.write(result_frame)
#         else:
#             predictor.fps_counter.run(True)
#             break


# def main(exp: Exp, config: SimpleNamespace):
#     """Main code for running the evaluation

#     Args:
#         exp: Experiment object containing model configuration
#         config: Evaluation configuration
#     """
#     if not config.experiment_name:
#         config.experiment_name = exp.exp_name

#     exp_dir = config.root_dir / exp.output_dir / config.experiment_name
#     exp_dir.mkdir(parents=True, exist_ok=True)

#     vis_dir = exp_dir / "vis_res"
#     vis_dir.mkdir(parents=True, exist_ok=True)

#     # if config.trt:
#     #     config.device = "gpu"

#     logger.info(f"Args: {config}")

#     if config.conf is not None:
#         exp.test_conf = config.conf
#     if config.nms is not None:
#         exp.nmsthre = config.nms
#     if config.tsize is not None:
#         exp.test_size = (config.tsize, config.tsize)

#     model = exp.get_model()
#     logger.info(f"Model summary: {get_model_info(model, exp.test_size)}")

#     if config.device == "gpu":
#         model.cuda()
#         if config.fp16:
#             model.half()  # to FP16
#     model.eval()

#     if not config.trt:
#         if config.ckpt is None:
#             ckpt_file = str(exp_dir / "best_ckpt.pth")
#         else:
#             ckpt_file = str(config.root_dir / config.ckpt)
#         logger.info("Loading checkpoint")
#         ckpt = torch.load(ckpt_file, map_location="cpu")
#         # load the model state dict
#         model.load_state_dict(ckpt["model"])
#         logger.info("Loaded checkpoint done.")

#     # if config.fuse:
#     #     logger.info("\tFusing model...")
#     #     model = fuse_model(model)

#     # if config.trt:
#     #     assert not config.fuse, "TensorRT model is not support model fusing!"
#     #     trt_file = exp_dir / "model_trt.pth"
#     #     assert (
#     #         trt_file.exists()
#     #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
#     #     model.head.decode_in_inference = False
#     #     decoder = model.head.decode_outputs
#     #     logger.info("Using TensorRT to inference")
#     # else:
#     #     trt_file = None
#     #     decoder = None
#     trt_file = None
#     decoder = None

#     predictor = Predictor(
#         model,
#         exp,
#         COCO_CLASSES,
#         trt_file,
#         decoder,
#         config.device,
#         config.fp16,
#         config.legacy,
#     )
#     current_time = time.localtime()
#     if config.mode == "image":
#         evaluate_image(predictor, vis_dir, config.path, current_time)
#     elif config.mode in ("video", "webcam"):
#         evaluate_imageflow(predictor, vis_dir, current_time, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    PROJ_DIR = Path(__file__).resolve().parent

    with open(PROJ_DIR / "eval_config.yml") as infile:
        CONFIG = yaml.safe_load(infile)
    CONFIG["root"] = PROJ_DIR
    # External configs
    CONFIG["input_dir"] = Path("~/code/YOLOX/data/video/MOT20-07-raw.mp4").expanduser()
    CONFIG["output_dir"] = PROJ_DIR / "YOLOX_outputs"
    # print(CONFIG)
    # EXP_OBJ = get_exp(CONFIG["name"])
    # print(EXP_OBJ.depth)
    # print(EXP_OBJ.width)

    yolox_node = YOLOXNode(CONFIG)
    image = np.array([0])
    inputs = {"img": image}
    yolox_node.run(inputs)

    # main(EXP_OBJ, CONFIG)
