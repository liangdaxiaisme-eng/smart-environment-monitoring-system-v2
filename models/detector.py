import os


def load_yolo_model(path):
    if not os.path.exists(path):
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        return model
    except Exception as e:
        print(f"[✗] YOLO模型加载失败: {e}")
        return None
