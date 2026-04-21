import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """交叉注意力多模态融合模型 — 综合环境评估"""
    def __init__(self, input_size=11, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.img_mlp = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=64, num_heads=1, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, temporal_input, image_features):
        lstm_out, _ = self.lstm(temporal_input)
        lstm_feat = lstm_out[:, -1, :]
        img_feat = self.img_mlp(image_features)
        query = img_feat.unsqueeze(1)
        key = lstm_feat.unsqueeze(1)
        value = lstm_feat.unsqueeze(1)
        attn_out, _ = self.cross_attn(query, key, value)
        attn_out = attn_out.squeeze(1)
        combined = torch.cat([lstm_feat, attn_out], dim=1)
        return torch.sigmoid(self.fc(combined))


def load_fusion_model(path, device):
    if not __import__('os').path.exists(path):
        return None
    try:
        model = FusionModel(input_size=11, hidden_size=64, num_layers=2)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"[✗] 融合模型加载失败: {e}")
        return None
