#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]} # 残差块数量：Darknet21中 
                                                        # dark2, dark3, dark4, dark5 层分别有 2、8、8、4 个残差块
                                                        # Darknet53中分别为2、8、8、4个

    def __init__(
        self,
        depth,                                      # 深度因子，有21和53可选，决定网络复杂程度
        in_channels=3,                              # 输入通道数
        stem_out_channels=32,                       # 初始部分输出通道数
        out_features=("dark3", "dark4", "dark5"),   # 输出通道名称列表
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()                          # 父类初始化
        assert out_features, "please provide output features of Darknet" # 断言，如果为假，则触发异常
        self.out_features = out_features

        # 网络初始部分
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),   # 基础卷积层，
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),          # 一系列层（输入通道数，块数量，步长）
        )                                           # Sequential()容器，让输入数据依次通过内部结构  

        # Dark2部分
        in_channels = stem_out_channels * 2         # in_channels变成64
        num_blocks = Darknet.depth2blocks[depth]    # 由深度因子和depth2blocks列表决定残差块数
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )

        # Dark3部分
        in_channels *= 2                            # in_channels变成128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )                                           # 层组合

        # Dark4部分
        in_channels *= 2                            # in_channels变成256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )                                           # 层组合

        # Dark5部分
        in_channels *= 2                            # in_channels变成512
        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),           # 层组合
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),  # 空间金字塔结构
        )                                           # 层组合                  

    # 层组合函数
    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),    # 基础卷积层
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],                      # 多个残差层
        ]
    
    # 空间金字塔函数
    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):   
        outputs = {}            # 创建空字典
        x = self.stem(x)        # 输入进入到初始部分
        outputs["stem"] = x     
        x = self.dark2(x)       # 进入Dark2部分
        outputs["dark2"] = x
        x = self.dark3(x)       # 进入Dark3部分
        outputs["dark3"] = x
        x = self.dark4(x)       # 进入Dark4部分
        outputs["dark4"] = x
        x = self.dark5(x)       # 进入Dark5部分
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}     # 返回仅包含"dark3", "dark4", "dark5"键值对的字典

# CSPDarknet结构
class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,            # 深度因子
        wid_mul,            # 宽度因子
        out_features=("dark3", "dark4", "dark5"),
        depthwise=True,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)        # 基础宽度64
        base_depth = max(round(dep_mul * 3), 1)  # 通过dep_mul决定网络深度

        # stem
        # self.stem = Focus(3, base_channels, ksize=3, depthwise=depthwise, act=act)

        # 初始部分
        self.stem = BaseConv(3, base_channels, ksize=6, stride=2, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            # CSP层
            CSPLayer(
                base_channels * 2,      # 输入宽度：基础宽度*4
                base_channels * 2,      # 输出宽度：基础宽度*4
                n=base_depth,           # bottleneck层数量
                depthwise=depthwise,    # 是否使用深度可分离卷积
                act=act,                # 激活函数
            ),                                  
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,               
                base_channels * 4,               
                n=base_depth * 3,                
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):                       # CSPDarknet结构的向前传播函数
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features} # 只返回了"dark3", "dark4", "dark5"的特征图
