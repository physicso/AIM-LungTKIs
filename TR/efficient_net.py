import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
 
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., nn.Module]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper & Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class MBConv(nn.Module):
    def __init__(
            self,
            cnf: MBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(nn.Module):
    def __init__(
            self,
            cnf: FusedMBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
            dropout: float,
            stochastic_depth_prob: float = 0.2,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            last_channel: Optional[int] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        #layers: List[nn.Module] = []
        self.layers =  nn.ModuleList([])
        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            self.layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        self.layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        #self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            

    def forward(self, x):
        feature = []
        for i, blk in enumerate(self.layers):
            x = blk(x)
            if (i + 1) % (len(self.layers) // 4) == 0:
                feature.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature.append(x)

        return feature


def _efficientnet_conf(
        arch: str,
        **kwargs: Any,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]

    
    if arch.startswith("efficientnet_v2_t"):

        inverted_residual_setting = [
        FusedMBConvConfig(1, 3, 1, 20, 20, 2),
        FusedMBConvConfig(4, 3, 2, 20, 40, 3),
        FusedMBConvConfig(4, 3, 2, 40, 56, 3),
        MBConvConfig(4, 3, 2, 56, 104, 4),
        MBConvConfig(6, 3, 1, 104, 120, 7),
        MBConvConfig(6, 3, 2, 120, 208, 10),
        ]
        last_channel = 512

    elif arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 640
    elif arch.startswith("efficientnet_v2_m"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


class Efficientv2_T(nn.Module):

    def __init__(self,Linear_only=False,num_class =2, pretrained=True):
        super().__init__()

        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_t")
        self.efficientnet = EfficientNet(inverted_residual_setting, dropout=0.1, last_channel=last_channel)
       
        self.Linear = nn.Linear(512, num_class)
        self.f = Linear_only
    def forward(self, x):
      
        feature = self.efficientnet(x)
        out = feature[-1]
       
        out = self.Linear(out)

        if self.f :
            return out
        else:
            return out,feature


class Efficientv2_S(nn.Module):

    def __init__(self,Linear_only=False,num_class =2, pretrained=True):
        super().__init__()

        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
        self.efficientnet = EfficientNet(inverted_residual_setting, dropout=0.1, last_channel=last_channel)
        if pretrained:
            self.efficientnet.load_state_dict(torch.load("./backbone_pretrained.pt", map_location=torch.device('cpu')))
        
        self.Linear = nn.Linear(640, num_class)
        self.f = Linear_only
    def forward(self, x):
      
        feature = self.efficientnet(x)
        out = feature[-1]
       
        out = self.Linear(out)

        if self.f :
            return out
        else:
            return out,feature


class Efficientv2_M(nn.Module):

    def __init__(self,Linear_only=False,num_class =2, pretrained=True):
        super().__init__()

        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
        self.efficientnet = EfficientNet(inverted_residual_setting, dropout=0.1, last_channel=last_channel)
 
        self.Linear = nn.Linear(1280, num_class)
        self.f = Linear_only


    def forward(self, x):
      
        feature = self.efficientnet(x)
        out = feature[-1]
       
        out = self.Linear(out)

        if self.f :
            return out
        else:
            return out,feature


class Efficientv2_L(nn.Module):

    def __init__(self,Linear_only=False,num_class =2, pretrained=True):
        super().__init__()

        inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
        self.efficientnet = EfficientNet(inverted_residual_setting, dropout=0.1, last_channel=last_channel)

        self.Linear = nn.Linear(1280, num_class)
        self.f = Linear_only
    def forward(self, x):
      
        feature = self.efficientnet(x)
        out = feature[-1]
       
        out = self.Linear(out)

        if self.f :
            return out
        else:
            return out,feature



if __name__ == '__main__':   
    
    net = Efficientv2_M(Linear_only=False,num_class=1024).cuda(7)
    
    a = torch.randn(size=(2, 3,256,256)).cuda(7)

    net.train()

    with torch.no_grad():
        net.zero_grad()
        out, feature  = net(a)
       # out = net(a)
        print(out[0].shape)
        print(feature[0].shape)
        print(feature[1].shape)
        print(feature[2].shape)
        print(feature[3].shape)
        print(feature[4].shape)