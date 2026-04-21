import torch
import torch.nn as nn
import torchvision.models as models


class SE(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1)


class ResNet18SE(nn.Module):
    """ResNet18 + SE 注意力 — 水质图像分类（4类）"""
    def __init__(self, num_classes=4):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.feat = nn.Sequential(*list(resnet.children())[:-2])
        self.se = SE(512, reduction=16)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.feat(x)
        x = self.se(x)
        x = x.mean([-1, -2])
        return self.fc(x)


def load_water_model(path, device):
    if not __import__('os').path.exists(path):
        return None
    try:
        model = ResNet18SE(num_classes=4)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"[✗] 水质模型加载失败: {e}")
        return None
