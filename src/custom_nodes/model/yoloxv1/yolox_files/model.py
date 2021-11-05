# Modifications copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Original copyright 2021 Megvii, Base Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""YOLOX model with its backbone (YOLOFAPN) and head (YOLOXHead).

Modifications include:
- YOLOX
    - Uses only YOLOPAFPN as backbone
    - Uses only YOLOHead as head
- YOLOPAFPN
    - Refactor arguments name
- YOLOXHead
    - Uses range based loop to iterate through in_channels
    - Removed training-related code and arguments
        - Code under the `if self.training` scope
        - get_output_and_grid() and initialize_biases() methods
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer

IN_CHANNELS = [256, 512, 1024]


class YOLOX(nn.Module):
    """YOLOX model module.
    The module list is defined by create_yolov3_modules function. The network
    returns loss values from three YOLO layers during training and detection
    results during test.
    """

    def __init__(
        self,
        num_classes: int,
        depth: float,
        width: float,
    ) -> None:
        super().__init__()
        self.backbone = YOLOPAFPN(depth, width)
        self.head = YOLOXHead(num_classes, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call"""
        # FPN output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(inputs)
        return self.head(fpn_outs)


class YOLOPAFPN(nn.Module):  # pylint: disable=too-many-instance-attributes
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    # pylint: disable=dangerous-default-value, invalid-name
    def __init__(
        self,
        depth: float = 1.0,
        width: float = 1.0,
    ):
        super().__init__()
        n_bottleneck = round(3 * depth)
        self.in_features = ("dark3", "dark4", "dark5")
        self.backbone = CSPDarknet(depth, width, self.in_features)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(IN_CHANNELS[2] * width), int(IN_CHANNELS[1] * width), 1, 1
        )
        self.C3_p4 = self.make_csp_layer(
            IN_CHANNELS[1], IN_CHANNELS[1], n_bottleneck, width
        )

        self.reduce_conv1 = BaseConv(
            int(IN_CHANNELS[1] * width), int(IN_CHANNELS[0] * width), 1, 1
        )
        self.C3_p3 = self.make_csp_layer(
            IN_CHANNELS[0], IN_CHANNELS[0], n_bottleneck, width
        )

        # bottom-up conv
        self.bu_conv2 = BaseConv(
            int(IN_CHANNELS[0] * width), int(IN_CHANNELS[0] * width), 3, 2
        )
        self.C3_n3 = self.make_csp_layer(
            IN_CHANNELS[0], IN_CHANNELS[1], n_bottleneck, width
        )

        # bottom-up conv
        self.bu_conv1 = BaseConv(
            int(IN_CHANNELS[1] * width), int(IN_CHANNELS[1] * width), 3, 2
        )
        self.C3_n4 = self.make_csp_layer(
            IN_CHANNELS[1], IN_CHANNELS[2], n_bottleneck, width
        )

    @staticmethod
    def make_csp_layer(
        in_channel: int, out_channel: int, depth: int, width: float
    ) -> CSPLayer:
        """Returns a CSPLayer"""
        return CSPLayer(
            int(2 * in_channel * width), int(out_channel * width), depth, False
        )

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        out_features = self.backbone(inputs)
        [x2, x1, x0] = [out_features[f] for f in self.in_features]

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Decoupled head.

    The decoupled head contains two parallel branches for classification and
    regression tasks. An IoU branch is added to the regression branch after
    the conv layers.
    """

    # pylint: disable=dangerous-default-value, invalid-name
    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        strides: List[int] = [8, 16, 32],
    ) -> None:
        """
        Args:
            act (str): activation type of conv. Default: "silu".
            depthwise (bool): Flag to determine whether to apply depthwise
                conv in conv branch. Default: False.
        """
        super().__init__()

        self.sizes: List[torch.Size]
        feat_channels = int(256 * width)
        self.n_anchors = 1
        self.num_classes = num_classes

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for in_channel in IN_CHANNELS:
            self.stems.append(BaseConv(int(in_channel * width), feat_channels, 1, 1))
            self.cls_convs.append(nn.Sequential(*self.make_group_layer(feat_channels)))
            self.reg_convs.append(nn.Sequential(*self.make_group_layer(feat_channels)))
            self.cls_preds.append(
                nn.Conv2d(feat_channels, self.n_anchors * self.num_classes, 1, 1, 0)
            )
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(feat_channels, self.n_anchors * 1, 1, 1, 0))

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(IN_CHANNELS)

    @staticmethod
    def make_group_layer(in_channels: int) -> Tuple[BaseConv, BaseConv]:
        """2x BaseConv layer"""
        return (
            BaseConv(in_channels, in_channels, 3, 1),
            BaseConv(in_channels, in_channels, 3, 1),
        )

    def forward(
        self, xin: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Defines the computation performed at every call"""
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, xin)
        ):
            x = self.stems[k](x)

            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            outputs.append(output)

        self.sizes = [x.shape[-2:] for x in outputs]
        # [batch, n_anchors_all, 85]
        outputs_tensor = torch.cat(
            [x.flatten(start_dim=2) for x in outputs], dim=2
        ).permute(0, 2, 1)
        # Always decode output for inference
        outputs_tensor = self.decode_outputs(outputs_tensor, xin[0].type())
        return outputs_tensor

    def decode_outputs(self, outputs: torch.Tensor, dtype: str) -> torch.Tensor:
        """Converts raw output to [x, y, w, h] format."""
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.sizes, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids_tensor = torch.cat(grids, dim=1).type(dtype)
        strides_tensor = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids_tensor) * strides_tensor
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides_tensor
        return outputs
