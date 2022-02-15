import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class MetricIntersection(nn.Module):
    def __init__(self):
        super(MetricIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = F.one_hot(gt, num_classes=num_classes)

        intersection = torch.logical_and(pred, gt)

        return torch.sum(intersection, dim=(1, 2))

class MetricSumIntersection(MetricIntersection):
    def __init__(self):
        super(MetricSumIntersection, self).__init__()

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt), dim=0)

class MetricUnion(nn.Module):
    def __init__(self, smooth: float = 1e-10):
        super(MetricUnion, self).__init__()
        self.smooth = torch.tensor(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        num_classes = pred.shape[1]

        pred = torch.argmax(pred, dim=1)
        pred = F.one_hot(pred, num_classes=num_classes)
        gt = F.one_hot(gt, num_classes=num_classes)

        union = torch.logical_or(pred, gt)

        return torch.sum(union, (1, 2)) + self.smooth

class MetricSumUnion(MetricUnion):
    def __init__(self, smooth: float = 1e-10):
        super().__init__(smooth)

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        return torch.sum(super().forward(pred, gt) - self.smooth, dim=0) + self.smooth
