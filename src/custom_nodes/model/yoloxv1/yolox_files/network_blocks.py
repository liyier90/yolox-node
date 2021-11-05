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

"""Network blocks for constructing the YOLOX model

Modifications include:
- Remove SiLU export-friendly class
- Removed ReLU and LeakyReLU activation functions
- ResLayer class
"""

from typing import Tuple

import torch
import torch.nn as nn


def get_activation(name: str = "silu", inplace: bool = True) -> nn.SiLU:
    """The activation function.

    Args:
        name (str): Name of the activation function. Default: "silu".
        inplace (bool): Flag for whether to do the operation in-place.
            Default: True

    Returns:
        (nn.Module): Activation function.

    Raises:
        AttributeError: When requesting an activation function other than
            "silu"
    """
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu block"""

    # pylint: disable=invalid-name, too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call"""
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x: torch.Tensor) -> torch.Tensor:
        """The computation performed at every call when conv and batch norm
        layers are fused.
        """
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    # pylint: disable=invalid-name, too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int = 1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call"""
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    # pylint: disable=invalid-name, too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call"""
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    # pylint: disable=invalid-name
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, int, int] = (5, 9, 13),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call"""
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    # pylint: disable=invalid-name, too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act: str = "silu",
    ) -> None:
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call"""
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    # pylint: disable=invalid-name, too-many-arguments
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 1,
        stride: int = 1,
        act: str = "silu",
    ) -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call"""
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
