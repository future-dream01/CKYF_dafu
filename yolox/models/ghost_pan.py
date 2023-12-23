# Copyright 2021 RangiLyu.
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
import torch
import torch.nn as nn

from .ghostnet import GhostBottleneck
# from ..module.conv import ConvModule, DepthwiseConvModule
from .network_blocks import BaseConv, DWConv


class GhostBlocks(nn.Module):
    """Stack of GhostBottleneck used in GhostPAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        activation (str): Name of activation function. Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        expand=1,
        kernel_size=5,
        num_blocks=1,
        use_res=False,
        activation="hswish",
    ):
        super(GhostBlocks, self).__init__()
        self.use_res = use_res
        if use_res:
            self.reduce_conv = BaseConv(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                act=activation,
            )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                GhostBottleneck(
                    in_channels,
                    int(out_channels * expand),
                    out_channels,
                    dw_kernel_size=kernel_size,
                    activation=activation,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out

# GhostPAN结构
#TODO:可尝试修改上采样配置
class GhostPAN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,  # 输出通道数分别为 256, 512, 1024
        use_depthwise=False,
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="hswish",
    ):
        super(GhostPAN, self).__init__()
        assert len(out_channels) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        Conv = DWConv if use_depthwise else BaseConv

        # 创建上采样、下采样和融合层
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        # 构建上采样层和融合层
        for idx in range(len(in_channels)):
            if idx < len(in_channels) - 1:
                self.reduce_layers.append(
                    BaseConv(
                        in_channels=in_channels[idx],
                        out_channels=out_channels[idx],
                        ksize=1, stride=1, act=self.activation,
                    )
                )
                self.top_down_blocks.append(
                    GhostBlocks(
                        out_channels[idx] + out_channels[idx + 1],
                        out_channels[idx],
                        expand, kernel_size, num_blocks, use_res, activation,
                    )
                )

        # 构建下采样层和融合层
        for idx in range(1, len(in_channels)):
            self.downsamples.append(
                Conv(
                    out_channels[idx - 1],
                    out_channels[idx - 1],
                    kernel_size, stride=2, act=self.activation,
                )
            )
            self.bottom_up_blocks.append(
                GhostBlocks(
                    out_channels[idx - 1] + out_channels[idx],
                    out_channels[idx],
                    expand, kernel_size, num_blocks, use_res, activation,
                )
            )

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 缩减层
        inputs = [reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)]

        # 自顶向下路径
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            upsample_feat = self.upsample(inner_outs[0])
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, inputs[idx - 1]], 1)
            )
            inner_outs.insert(0, inner_out)

        # 自底向上路径
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            downsample_feat = self.downsamples[idx](outs[-1])
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, inner_outs[idx + 1]], 1)
            )
            outs.append(out)

        return tuple(outs)