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

"""A collection of utility functions used by YOLOX

Modifications include:
- Refactor fuse_conv_and_bn and fuse_model
- Adding xywh2xyxy and xyxy2xyxyn
"""

import numpy as np
import torch
import torch.nn as nn

from .network_blocks import BaseConv


def fuse_conv_and_bn(conv, batch_norm):
    """Fuses convolution and batchnorm layers.
    Reference: https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(
        batch_norm.weight.div(torch.sqrt(batch_norm.eps + batch_norm.running_var))
    )
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = batch_norm.bias - batch_norm.weight.mul(batch_norm.running_mean).div(
        torch.sqrt(batch_norm.running_var + batch_norm.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    """Fuses the batch normalization layers in `BaseConv` modules."""
    for module in model.modules():
        if isinstance(module, BaseConv) and hasattr(module, "bn"):
            module.conv = fuse_conv_and_bn(module.conv, module.bn)  # update conv
            delattr(module, "bn")  # remove batchnorm
            module.forward = module.fuseforward  # update forward
    return model


def xywh2xyxy(inputs: torch.Tensor) -> torch.Tensor:
    """Converts from [x, y, w, h] to [x1, y1, x2, y2] format.

    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
    """
    outputs = torch.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] - inputs[:, 2] / 2
    outputs[:, 1] = inputs[:, 1] - inputs[:, 3] / 2
    outputs[:, 2] = inputs[:, 0] + inputs[:, 2] / 2
    outputs[:, 3] = inputs[:, 1] + inputs[:, 3] / 2

    return outputs


def xyxy2xyxyn(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from [x1, y1, x2, y2] to normalized [x1, y1, x2, y2].

    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
    Normalized coordinates are w.r.t. original image size.
    """
    outputs = np.empty_like(inputs)
    outputs[:, [0, 2]] = inputs[:, [0, 2]] / width
    outputs[:, [1, 3]] = inputs[:, [1, 3]] / height

    return outputs
