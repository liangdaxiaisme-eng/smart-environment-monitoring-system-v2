import torch
import torch.nn as nn


class LSTM_Predictor(nn.Module):
    """LSTM + 时序注意力 — PM2.5 预测"""
    def __init__(self, input_size=11, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


def load_air_model(path, device):
    if not __import__('os').path.exists(path):
        return None
    try:
        model = LSTM_Predictor(input_size=11, hidden_size=128, num_layers=2)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"[✗] 空气模型加载失败: {e}")
        return None
