"""Spectrogram → per-frame token log-probabilities, trained with CTC.

The same shape of network as DeepFist (written from scratch): a small 2D convolutional stem that
shrinks the frequency axis, a max over frequency (so the tone's pitch doesn't matter), then
dilated 1D residual convolutions for timing context (~4 s receptive field), and a 1×1
classifier. About 0.5M parameters. Frames are 16 ms after the stem halves the time axis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cwsynth.alphabet import TOKENS

N_CLASSES = len(TOKENS)
TIME_STRIDE = 2


class ConvBlock2d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int]):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResidualTemporal(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.bn(self.conv(x)))


class CwNet(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, channels: int = 128,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 1)):
        super().__init__()
        # Strides are (frequency, time): 33 bins → 17 → 9 → 5, time halved once.
        self.stem = nn.Sequential(
            ConvBlock2d(1, 32, (1, 1)),
            ConvBlock2d(32, 48, (2, 1)),
            ConvBlock2d(48, 64, (2, TIME_STRIDE)),
            ConvBlock2d(64, 96, (2, 1)),
        )
        self.project = nn.Sequential(nn.Conv1d(96, channels, 1, bias=False),
                                     nn.BatchNorm1d(channels), nn.ReLU())
        self.temporal = nn.Sequential(*(ResidualTemporal(channels, d) for d in dilations))
        self.head = nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False),
                                  nn.BatchNorm1d(channels), nn.ReLU(),
                                  nn.Conv1d(channels, n_classes, 1))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """[batch, 1, bins, frames] → log-probabilities [frames // 2 (rounded up), batch, classes]."""
        x = self.stem(spec).amax(dim=2)
        x = self.head(self.temporal(self.project(x)))
        return F.log_softmax(x.permute(2, 0, 1).float(), dim=-1)


def output_lengths(input_frames: torch.Tensor) -> torch.Tensor:
    """Frames out of the stem for each input length (kernel 3, padding 1, stride 2)."""
    return (input_frames - 1) // TIME_STRIDE + 1


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
