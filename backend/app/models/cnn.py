"""
BovineVision AI - Original Configurable CNN Architecture
Designed and trained entirely from scratch without pretrained weights or transfer learning.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention module.
    Recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: Global Average Pooling (B, C, H, W) -> (B, C, 1, 1)
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        # Excitation
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        # Recalibrate
        return x * scale


class BovineResidualBlock(nn.Module):
    """
    Configurable residual convolutional block featuring:
    - 3x3 conv -> BatchNorm -> GELU -> 3x3 conv -> BatchNorm
    - Optional Squeeze-and-Excitation channel recalibration
    - Identity/Projection skip connection
    - Dropout regularization
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SqueezeExcitationBlock(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Shortcut projection if dimension or stride changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.act_out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)
        out = self.dropout(out)

        out = out + identity
        out = self.act_out(out)
        return out


class BovineVisionCNN(nn.Module):
    """
    Original, Configurable Deep Convolutional Neural Network for cattle breed classification.
    Parameterized with:
      - in_channels: input image channels (default 3)
      - num_classes: target cattle breeds
      - stem_channels: initial convolutional stem depth
      - stage_channels: channel depth per hierarchical stage
      - blocks_per_stage: number of residual blocks per stage
      - strides: downsampling stride per stage
      - use_se: whether to activate channel attention
      - dropout_rate: regularization factor
      - classifier_hidden: width of dense classification head
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        stem_channels: int = 32,
        stage_channels: Optional[List[int]] = None,
        blocks_per_stage: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        use_se: bool = True,
        dropout_rate: float = 0.25,
        classifier_hidden: int = 256
    ):
        super().__init__()

        if stage_channels is None:
            stage_channels = [32, 64, 128, 256]
        if blocks_per_stage is None:
            blocks_per_stage = [2, 2, 2, 2]
        if strides is None:
            strides = [1, 2, 2, 2]

        self.config = {
            "architecture": "BovineVisionCNN-v1",
            "in_channels": in_channels,
            "num_classes": num_classes,
            "stem_channels": stem_channels,
            "stage_channels": stage_channels,
            "blocks_per_stage": blocks_per_stage,
            "strides": strides,
            "use_se": use_se,
            "dropout_rate": dropout_rate,
            "classifier_hidden": classifier_hidden
        }

        # Initial Feature Stem: progressive downsampling with preservation of fine spatial cues
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Hierarchical Stages
        stages = []
        current_in = stem_channels
        for stage_idx, (out_c, num_blocks, stride) in enumerate(
            zip(stage_channels, blocks_per_stage, strides)
        ):
            stage_layers = []
            for block_idx in range(num_blocks):
                b_stride = stride if block_idx == 0 else 1
                stage_layers.append(
                    BovineResidualBlock(
                        in_channels=current_in,
                        out_channels=out_c,
                        stride=b_stride,
                        use_se=use_se,
                        dropout_rate=dropout_rate if block_idx == num_blocks - 1 else 0.0
                    )
                )
                current_in = out_c
            stages.append(nn.Sequential(*stage_layers))

        self.stages = nn.ModuleList(stages)
        self.final_features = current_in

        # Global Pooling and Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.final_features, classifier_hidden, bias=False),
            nn.BatchNorm1d(classifier_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.8),
            nn.Linear(classifier_hidden, num_classes)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.global_pool(out)
        logits = self.classifier(out)
        return logits

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Provides layer structure and parameter statistics for inspection."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        stage_info = []
        for idx, (ch, num_b) in enumerate(zip(self.config["stage_channels"], self.config["blocks_per_stage"])):
            stage_info.append({
                "stage": idx + 1,
                "channels": ch,
                "blocks": num_b,
                "attention": "Squeeze-and-Excitation (SE)" if self.config["use_se"] else "None"
            })

        return {
            "name": "BovineVisionCNN",
            "type": "Original Custom Deep Residual CNN with SE-Attention",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_resolution": "224x224x3",
            "num_classes": self.config["num_classes"],
            "stages": stage_info,
            "stem": f"Conv3x3 (stride 2) -> BN -> GELU -> Conv3x3 -> MaxPool (Channels: {self.config['stem_channels']})",
            "classification_head": f"AdaptiveAvgPool2d -> Linear({self.final_features}->{self.config['classifier_hidden']}) -> BN -> GELU -> Linear({self.config['classifier_hidden']}->{self.config['num_classes']})",
            "pretrained": False,
            "weights_origin": "Trained entirely from scratch on Mendeley Dataset xc3j9v382w"
        }
