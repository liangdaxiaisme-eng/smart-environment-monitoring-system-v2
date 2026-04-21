import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


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


class GradCAM:
    """ResNet18+SE 的 Grad-CAM 热力图生成器"""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # 注册 hook：捕获 feat 模块输出（最后一次卷积的 feature map）
        self.model.feat.register_forward_hook(self._forward_hook)
        self.model.feat.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 热力图
        Args:
            input_tensor: [1, 3, 224, 224] 预处理后的图像张量
            target_class: 目标类别索引，默认取预测类别
        Returns:
            heatmap: [224, 224] numpy, 归一化到 [0, 1]
        """
        self.model.eval()
        # 前向传播
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播目标类别分数
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM: 对梯度做全局平均池化 → 加权求和 feature map
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, 512, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, 7, 7]
        cam = torch.relu(cam)

        # 上采样到输入尺寸 + 归一化
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam

    @staticmethod
    def overlay_heatmap(original_img, heatmap, alpha=0.45):
        """
        将热力图叠加到原图上
        Args:
            original_img: PIL Image (RGB)
            heatmap: [H, W] numpy, [0, 1]
            alpha: 热力图透明度
        Returns:
            overlay: PIL Image (RGB)
        """
        from PIL import Image as PILImage
        import matplotlib.cm as cm

        # 应用 JET colormap
        colormap = cm.get_cmap('jet')
        colored = colormap(heatmap)[:, :, :3]  # [H, W, 3], float64
        colored = (colored * 255).astype(np.uint8)
        heatmap_img = PILImage.fromarray(colored)

        # 调整原图尺寸一致
        original_resized = original_img.resize((224, 224), PILImage.LANCZOS)

        # alpha 混合
        overlay = PILImage.blend(original_resized.convert('RGB'), heatmap_img, alpha)
        return overlay
