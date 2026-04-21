# 智能环境监测系统 v2

基于深度学习的环境监测与预测平台 · 全新 UI 重构版

## 功能模块

| 模块 | 功能 | 模型 | 性能 |
|------|------|------|------|
| 💧 水质分析 | 图像分类识别污染等级 | ResNet18 + SE Attention | 准确率 90.8% |
| 🌫️ 空气预测 | PM2.5 时序预测 | LSTM + Temporal Attention | R²=0.94, RMSE=8.7 |
| 🚮 垃圾检测 | 河面目标检测（含标注图） | YOLO11n | mAP50=93.3% |
| 🔗 综合评估 | 多模态融合评估 | Cross-Attention Fusion | 准确率 94% |

## 快速部署

```bash
pip install -r requirements.txt
python app.py
# 访问 http://localhost:5000
```

## 项目结构

```
v2/
├── app.py              # 主应用（Flask + 推理逻辑 + UI 模板）
├── models/             # 模型定义（独立模块化）
│   ├── water.py        # ResNet18+SE 水质分类
│   ├── air.py          # LSTM+Attention 空气预测
│   ├── fusion.py       # Cross-Attention 融合模型
│   └── detector.py     # YOLO 垃圾检测
├── weights/            # 预训练权重文件
├── static/
│   ├── uploads/        # 上传/标注图临时目录
│   └── preview.html    # UI 静态预览
└── requirements.txt
```

## v2 改进

- 🎨 深色主题 UI，侧边栏导航，卡片式布局
- 📊 Dashboard 系统总览，模型性能对比
- 🖼️ 垃圾检测支持标注图可视化（边界框+标签+置信度）
- 📦 模型定义模块化，代码结构清晰

## License

MIT
