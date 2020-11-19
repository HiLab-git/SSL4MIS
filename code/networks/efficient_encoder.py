import re
from typing import List

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_model_params, url_map
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self) -> List:
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution chennels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, in_channels=in_channels)


def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()


class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = list(stage_idxs) + [len(self._blocks)]
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc

    def forward(self, x):

        features = [x]

        if self._depth > 0:
            x = self._swish(self._bn0(self._conv_stem(x)))
            features.append(x)

        if self._depth > 1:
            skip_connection_idx = 0
            for idx, block in enumerate(self._blocks):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self._blocks)
                x = block(x, drop_connect_rate=drop_connect_rate)
                if idx == self._stage_idxs[skip_connection_idx] - 1:
                    skip_connection_idx += 1
                    features.append(x)
                    if skip_connection_idx + 1 == self._depth:
                        break
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias")
        state_dict.pop("_fc.weight")
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        }
    }
    return pretrained_settings


efficient_net_encoders = {
    "efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b0"),
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (3, 5, 9),
            "model_name": "efficientnet-b0",
        },
    },
    "efficientnet-b1": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b1"),
        "params": {
            "out_channels": (3, 32, 24, 40, 112, 320),
            "stage_idxs": (5, 8, 16),
            "model_name": "efficientnet-b1",
        },
    },
    "efficientnet-b2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b2"),
        "params": {
            "out_channels": (3, 32, 24, 48, 120, 352),
            "stage_idxs": (5, 8, 16),
            "model_name": "efficientnet-b2",
        },
    },
    "efficientnet-b3": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b3"),
        "params": {
            "out_channels": (3, 40, 32, 48, 136, 384),
            "stage_idxs": (5, 8, 18),
            "model_name": "efficientnet-b3",
        },
    },
    "efficientnet-b4": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b4"),
        "params": {
            "out_channels": (3, 48, 32, 56, 160, 448),
            "stage_idxs": (6, 10, 22),
            "model_name": "efficientnet-b4",
        },
    },
    "efficientnet-b5": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b5"),
        "params": {
            "out_channels": (3, 48, 40, 64, 176, 512),
            "stage_idxs": (8, 13, 27),
            "model_name": "efficientnet-b5",
        },
    },
    "efficientnet-b6": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b6"),
        "params": {
            "out_channels": (3, 56, 40, 72, 200, 576),
            "stage_idxs": (9, 15, 31),
            "model_name": "efficientnet-b6",
        },
    },
    "efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b7"),
        "params": {
            "out_channels": (3, 64, 48, 80, 224, 640),
            "stage_idxs": (11, 18, 38),
            "model_name": "efficientnet-b7",
        },
    },
}

encoders = {}
encoders.update(efficient_net_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None):
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder



# class ResNetEncoder(ResNet, EncoderMixin):
#     def __init__(self, out_channels, depth=5, **kwargs):
#         super().__init__(**kwargs)
#         self._depth = depth
#         self._out_channels = out_channels
#         self._in_channels = 3

#         del self.fc
#         del self.avgpool

#     def get_stages(self):
#         return [
#             nn.Identity(),
#             nn.Sequential(self.conv1, self.bn1, self.relu),
#             nn.Sequential(self.maxpool, self.layer1),
#             self.layer2,
#             self.layer3,
#             self.layer4,
#         ]

#     def forward(self, x):
#         stages = self.get_stages()

#         features = []
#         for i in range(self._depth + 1):
#             x = stages[i](x)
#             features.append(x)

#         return features

#     def load_state_dict(self, state_dict, **kwargs):
#         state_dict.pop("fc.bias")
#         state_dict.pop("fc.weight")
#         super().load_state_dict(state_dict, **kwargs)


# resnet_encoders = {
#     "resnet18": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet18"],
#         "params": {
#             "out_channels": (3, 64, 64, 128, 256, 512),
#             "block": BasicBlock,
#             "layers": [2, 2, 2, 2],
#         },
#     },
#     "resnet34": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet34"],
#         "params": {
#             "out_channels": (3, 64, 64, 128, 256, 512),
#             "block": BasicBlock,
#             "layers": [3, 4, 6, 3],
#         },
#     },
#     "resnet50": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet50"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 6, 3],
#         },
#     },
#     "resnet101": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet101"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#         },
#     },
#     "resnet152": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": pretrained_settings["resnet152"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 8, 36, 3],
#         },
#     },
#     "resnext50_32x4d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": {
#             "imagenet": {
#                 "url": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
#                 "input_space": "RGB",
#                 "input_size": [3, 224, 224],
#                 "input_range": [0, 1],
#                 "mean": [0.485, 0.456, 0.406],
#                 "std": [0.229, 0.224, 0.225],
#                 "num_classes": 1000,
#             }
#         },
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 6, 3],
#             "groups": 32,
#             "width_per_group": 4,
#         },
#     },
#     "resnext101_32x8d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": {
#             "imagenet": {
#                 "url": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
#                 "input_space": "RGB",
#                 "input_size": [3, 224, 224],
#                 "input_range": [0, 1],
#                 "mean": [0.485, 0.456, 0.406],
#                 "std": [0.229, 0.224, 0.225],
#                 "num_classes": 1000,
#             },
#             "instagram": {
#                 "url": "https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth",
#                 "input_space": "RGB",
#                 "input_size": [3, 224, 224],
#                 "input_range": [0, 1],
#                 "mean": [0.485, 0.456, 0.406],
#                 "std": [0.229, 0.224, 0.225],
#                 "num_classes": 1000,
#             },
#         },
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 8,
#         },
#     },
#     "resnext101_32x16d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": {
#             "instagram": {
#                 "url": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
#                 "input_space": "RGB",
#                 "input_size": [3, 224, 224],
#                 "input_range": [0, 1],
#                 "mean": [0.485, 0.456, 0.406],
#                 "std": [0.229, 0.224, 0.225],
#                 "num_classes": 1000,
#             }
#         },
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 16,
#         },
#     },
#     "resnext101_32x32d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": {
#             "instagram": {
#                 "url": "https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth",
#                 "input_space": "RGB",
#                 "input_size": [3, 224, 224],
#                 "input_range": [0, 1],
#                 "mean": [0.485, 0.456, 0.406],
#                 "std": [0.229, 0.224, 0.225],
#                 "num_classes": 1000,
#             }
#         },
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 32,
#         },
#     },
#     "resnext101_32x48d": {
#         "encoder": ResNetEncoder,
#         "pretrained_settings": {
#             "instagram": {
#                 "url": "https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth",
#                 "input_space": "RGB",
#                 "input_size": [3, 224, 224],
#                 "input_range": [0, 1],
#                 "mean": [0.485, 0.456, 0.406],
#                 "std": [0.229, 0.224, 0.225],
#                 "num_classes": 1000,
#             }
#         },
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 2048),
#             "block": Bottleneck,
#             "layers": [3, 4, 23, 3],
#             "groups": 32,
#             "width_per_group": 48,
#         },
#     },
# }


# class TransitionWithSkip(nn.Module):

#     def __init__(self, module):
#         super().__init__()
#         self.module = module

#     def forward(self, x):
#         for module in self.module:
#             x = module(x)
#             if isinstance(module, nn.ReLU):
#                 skip = x
#         return x, skip


# class DenseNetEncoder(DenseNet, EncoderMixin):
#     def __init__(self, out_channels, depth=5, **kwargs):
#         super().__init__(**kwargs)
#         self._out_channels = out_channels
#         self._depth = depth
#         self._in_channels = 3
#         del self.classifier

#     def make_dilated(self, stage_list, dilation_list):
#         raise ValueError("DenseNet encoders do not support dilated mode "
#                          "due to pooling operation for downsampling!")

#     def get_stages(self):
#         return [
#             nn.Identity(),
#             nn.Sequential(self.features.conv0, self.features.norm0, self.features.relu0),
#             nn.Sequential(self.features.pool0, self.features.denseblock1,
#                           TransitionWithSkip(self.features.transition1)),
#             nn.Sequential(self.features.denseblock2, TransitionWithSkip(self.features.transition2)),
#             nn.Sequential(self.features.denseblock3, TransitionWithSkip(self.features.transition3)),
#             nn.Sequential(self.features.denseblock4, self.features.norm5)
#         ]

#     def forward(self, x):

#         stages = self.get_stages()

#         features = []
#         for i in range(self._depth + 1):
#             x = stages[i](x)
#             if isinstance(x, (list, tuple)):
#                 x, skip = x
#                 features.append(skip)
#             else:
#                 features.append(x)

#         return features

#     def load_state_dict(self, state_dict):
#         pattern = re.compile(
#             r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
#         )
#         for key in list(state_dict.keys()):
#             res = pattern.match(key)
#             if res:
#                 new_key = res.group(1) + res.group(2)
#                 state_dict[new_key] = state_dict[key]
#                 del state_dict[key]

#         # remove linear
#         state_dict.pop("classifier.bias")
#         state_dict.pop("classifier.weight")

#         super().load_state_dict(state_dict)


# densenet_encoders = {
#     "densenet121": {
#         "encoder": DenseNetEncoder,
#         "pretrained_settings": pretrained_settings["densenet121"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1024, 1024),
#             "num_init_features": 64,
#             "growth_rate": 32,
#             "block_config": (6, 12, 24, 16),
#         },
#     },
#     "densenet169": {
#         "encoder": DenseNetEncoder,
#         "pretrained_settings": pretrained_settings["densenet169"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1280, 1664),
#             "num_init_features": 64,
#             "growth_rate": 32,
#             "block_config": (6, 12, 32, 32),
#         },
#     },
#     "densenet201": {
#         "encoder": DenseNetEncoder,
#         "pretrained_settings": pretrained_settings["densenet201"],
#         "params": {
#             "out_channels": (3, 64, 256, 512, 1792, 1920),
#             "num_init_features": 64,
#             "growth_rate": 32,
#             "block_config": (6, 12, 48, 32),
#         },
#     },
#     "densenet161": {
#         "encoder": DenseNetEncoder,
#         "pretrained_settings": pretrained_settings["densenet161"],
#         "params": {
#             "out_channels": (3, 96, 384, 768, 2112, 2208),
#             "num_init_features": 96,
#             "growth_rate": 48,
#             "block_config": (6, 12, 36, 24),
#         },
#     },
# }
# net = get_encoder(name="efficientnet-b4", in_channels=1, depth=5, weights="imagenet")
#
# t = torch.rand(2, 1, 480, 480)
#
# print(len(net(t)))
