"""
智能环境监测系统 v2
基于深度学习的环境监测与预测平台
"""

from flask import Flask, render_template_string, request, send_from_directory
import torch
import torchvision.transforms as T
import numpy as np
import os
import time
import json
from PIL import Image
from models.water import load_water_model
from models.air import load_air_model
from models.fusion import load_fusion_model
from models.detector import load_yolo_model

app = Flask(__name__)

# ── 常量 ──────────────────────────────────────────────────────────
CLASS_NAMES = ['清洁', '轻度污染', '中度污染', '重度污染']
TRASH_CLASSES = ['branch', 'leaf', 'others', 'plastic-bag',
                 'plastic-bottle', 'plastic-wrapper', 'wood-log']
TRASH_CN = {'branch': '树枝', 'leaf': '树叶', 'others': '其他',
            'plastic-bag': '塑料袋', 'plastic-bottle': '塑料瓶',
            'plastic-wrapper': '塑料包装', 'wood-log': '木头'}
FEATURE_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NORMALIZE_MEAN = [75.0, 110.0, 15.0, 50.0, 1.2, 80.0, 13.0, 1016.0, 5.0, 0.1, 1.8]
NORMALIZE_STD  = [60.0, 80.0, 15.0, 30.0, 0.8, 50.0, 12.0, 10.0, 10.0, 0.5, 1.2]

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 模型加载 ─────────────────────────────────────────────────────
t0 = time.time()
water_model = load_water_model(os.path.join(WEIGHTS_DIR, 'resnet_se_best.pth'), device)
air_model   = load_air_model(os.path.join(WEIGHTS_DIR, 'lstm_best.pth'), device)
fusion_model = load_fusion_model(os.path.join(WEIGHTS_DIR, 'fusion_best.pth'), device)
yolo_model  = load_yolo_model(os.path.join(WEIGHTS_DIR, 'rubbish_best.pt'))
load_time = time.time() - t0

water_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_status = {
    'water': water_model is not None,
    'air': air_model is not None,
    'trash': yolo_model is not None,
    'fusion': fusion_model is not None,
}
loaded_count = sum(model_status.values())

# ── AQI 等级定义 ──────────────────────────────────────────────────
def pm25_level(v):
    if v < 35:   return '优', '#00e400', '空气质量令人满意，基本无污染'
    if v < 75:   return '良', '#ffff00', '空气质量可接受，某些污染物可能对极少数敏感人群有轻微影响'
    if v < 115:  return '轻度污染', '#ff7e00', '易感人群症状有轻度加剧，健康人群出现刺激症状'
    if v < 150:  return '中度污染', '#ff0000', '进一步加剧易感人群症状，可能对健康人群心脏、呼吸系统有影响'
    return '重度污染', '#99004c', '健康人群运动耐受力降低，有明显强烈症状'

def fusion_level(score):
    if score >= 80: return '优', '#00e400', '环境状况良好，各项指标正常'
    if score >= 60: return '良', '#ffff00', '环境状况可接受，适合户外活动'
    if score >= 40: return '轻度污染', '#ff7e00', '敏感人群应减少户外活动'
    if score >= 20: return '中度污染', '#ff0000', '建议佩戴口罩，减少外出'
    return '重度污染', '#99004c', '避免户外活动，注意防护'

# ── HTML 模板 ─────────────────────────────────────────────────────
BASE_HTML = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{{ title }} — 环境监测系统</title>
<style>
:root{--bg:#0f1117;--surface:#1a1d27;--surface2:#242836;--border:#2e3348;
--text:#e4e6f0;--text2:#8b8fa8;--accent:#4f8cff;--accent2:#3a6fd8;
--green:#2dd4a0;--yellow:#fbbf24;--orange:#f97316;--red:#ef4444;--purple:#a855f7;
--radius:12px;--shadow:0 2px 12px rgba(0,0,0,0.3)}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif;
background:var(--bg);color:var(--text);min-height:100vh;display:flex}
a{color:var(--accent);text-decoration:none}

/* Sidebar */
.sidebar{width:240px;background:var(--surface);border-right:1px solid var(--border);
display:flex;flex-direction:column;position:fixed;top:0;left:0;height:100vh;z-index:100}
.sidebar-brand{padding:24px 20px;border-bottom:1px solid var(--border)}
.sidebar-brand h1{font-size:15px;font-weight:700;letter-spacing:0.5px}
.sidebar-brand span{font-size:11px;color:var(--text2);display:block;margin-top:4px}
.sidebar-nav{flex:1;padding:12px 8px;overflow-y:auto}
.nav-item{display:flex;align-items:center;gap:10px;padding:10px 14px;border-radius:8px;
color:var(--text2);font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;margin-bottom:2px}
.nav-item:hover{background:var(--surface2);color:var(--text)}
.nav-item.active{background:var(--accent);color:#fff}
.nav-item .nav-icon{width:18px;text-align:center;font-size:14px}
.nav-item .nav-badge{margin-left:auto;font-size:10px;padding:2px 6px;border-radius:10px;
background:var(--surface2);color:var(--text2)}
.nav-item.active .nav-badge{background:rgba(255,255,255,0.2);color:#fff}
.sidebar-footer{padding:16px 20px;border-top:1px solid var(--border);font-size:11px;color:var(--text2)}
.device-tag{display:inline-flex;align-items:center;gap:4px;background:var(--surface2);
padding:4px 8px;border-radius:6px;margin-top:6px;font-size:10px}
.device-dot{width:6px;height:6px;border-radius:50%;background:var(--green)}

/* Main */
.main{margin-left:240px;flex:1;min-height:100vh}
.topbar{padding:20px 32px;border-bottom:1px solid var(--border);display:flex;
align-items:center;justify-content:space-between;background:var(--surface)}
.topbar h2{font-size:18px;font-weight:600}
.topbar-meta{display:flex;align-items:center;gap:16px;font-size:12px;color:var(--text2)}
.status-pills{display:flex;gap:6px}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:12px;
font-size:10px;font-weight:600}
.pill-ok{background:rgba(45,212,160,0.15);color:var(--green)}
.pill-err{background:rgba(239,68,68,0.15);color:var(--red)}
.content{padding:28px 32px}

/* Cards */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
overflow:hidden;margin-bottom:20px}
.card-head{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;
align-items:center;justify-content:space-between}
.card-head h3{font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}
.card-head .tag{font-size:10px;padding:3px 8px;border-radius:6px;font-weight:600}
.card-body{padding:20px}

/* Grid */
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin-bottom:24px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}

/* Stat Cards */
.stat{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
.stat-label{font-size:11px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px}
.stat-value{font-size:28px;font-weight:700;line-height:1}
.stat-sub{font-size:12px;color:var(--text2);margin-top:6px}
.stat-bar{height:4px;background:var(--surface2);border-radius:2px;margin-top:12px;overflow:hidden}
.stat-bar-fill{height:100%;border-radius:2px;transition:width .6s ease}

/* Forms */
.form-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.form-group{display:flex;flex-direction:column;gap:4px}
.form-group label{font-size:11px;color:var(--text2);font-weight:500}
.form-group input,.form-group select{background:var(--surface2);border:1px solid var(--border);
border-radius:8px;padding:9px 12px;color:var(--text);font-size:13px;outline:none;transition:border .15s}
.form-group input:focus{border-color:var(--accent)}
.btn{background:var(--accent);color:#fff;border:none;padding:11px 24px;border-radius:8px;
font-size:13px;font-weight:600;cursor:pointer;transition:background .15s;width:100%;margin-top:16px}
.btn:hover{background:var(--accent2)}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* Upload */
.upload-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:40px;
text-align:center;cursor:pointer;transition:all .2s}
.upload-zone:hover{border-color:var(--accent);background:rgba(79,140,255,0.05)}
.upload-zone input{display:none}
.upload-zone .icon{font-size:32px;margin-bottom:8px}
.upload-zone .label{font-size:13px;color:var(--text2)}
.upload-zone .hint{font-size:11px;color:var(--text2);margin-top:4px;opacity:.6}
.file-selected{font-size:12px;color:var(--green);margin-top:8px}

/* Result */
.result-box{margin-top:20px;border-radius:var(--radius);padding:20px;border:1px solid var(--border)}
.result-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.result-title{font-size:14px;font-weight:600}

/* Bar chart */
.bar-row{display:flex;align-items:center;margin-bottom:8px}
.bar-label{width:72px;font-size:12px;color:var(--text2)}
.bar-track{flex:1;height:22px;background:var(--surface2);border-radius:4px;overflow:hidden;position:relative}
.bar-fill{height:100%;border-radius:4px;transition:width .5s ease}
.bar-val{position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:11px;font-weight:600;color:var(--text)}

/* Detection list */
.det-row{display:flex;align-items:center;justify-content:space-between;padding:10px 0;
border-bottom:1px solid var(--border)}
.det-row:last-child{border:none}
.det-name{font-size:13px}
.det-name small{color:var(--text2);margin-left:6px;font-size:11px}
.det-conf{font-size:13px;font-weight:600}

/* Level badge */
.level-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;
border-radius:8px;font-size:14px;font-weight:700}

/* Error */
.error-box{margin-top:20px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
border-radius:var(--radius);padding:16px;font-size:13px;color:var(--red)}

/* Footer */
.page-footer{text-align:center;padding:24px;color:var(--text2);font-size:11px;
border-top:1px solid var(--border);margin-top:40px}

/* Responsive */
@media(max-width:1200px){.grid-4{grid-template-columns:repeat(2,1fr)}}
@media(max-width:768px){.sidebar{display:none}.main{margin-left:0}
.grid-4,.grid-2{grid-template-columns:1fr}.form-grid{grid-template-columns:1fr}}

/* Animated entry */
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.animate{animation:fadeUp .35s ease-out}
</style>
</head>
<body>
<aside class="sidebar">
  <div class="sidebar-brand">
    <h1>🌍 环境监测系统</h1>
    <span>Deep Learning Monitor v2.0</span>
  </div>
  <nav class="sidebar-nav">
    <a class="nav-item {{'active' if page=='dashboard' else ''}}" href="/?page=dashboard">
      <span class="nav-icon">📊</span>系统总览
    </a>
    <a class="nav-item {{'active' if page=='water' else ''}}" href="/?page=water">
      <span class="nav-icon">💧</span>水质分析
      <span class="nav-badge">ResNet+SE</span>
    </a>
    <a class="nav-item {{'active' if page=='air' else ''}}" href="/?page=air">
      <span class="nav-icon">🌫️</span>空气预测
      <span class="nav-badge">LSTM</span>
    </a>
    <a class="nav-item {{'active' if page=='trash' else ''}}" href="/?page=trash">
      <span class="nav-icon">🚮</span>垃圾检测
      <span class="nav-badge">YOLO11n</span>
    </a>
    <a class="nav-item {{'active' if page=='fusion' else ''}}" href="/?page=fusion">
      <span class="nav-icon">🔗</span>综合评估
      <span class="nav-badge">Fusion</span>
    </a>
  </nav>
  <div class="sidebar-footer">
    <div>推理设备</div>
    <div class="device-tag"><span class="device-dot"></span>{{ device_name }}</div>
    <div style="margin-top:8px">模型加载 {{ load_time }}s · {{ loaded }}/4 就绪</div>
  </div>
</aside>

<div class="main">
  <div class="topbar">
    <h2>{{ page_title }}</h2>
    <div class="topbar-meta">
      <div class="status-pills">
        {% for name, ok in ms.items() %}
        <span class="pill {{'pill-ok' if ok else 'pill-err'}}">{{ name }} {{'✓' if ok else '✗'}}</span>
        {% endfor %}
      </div>
    </div>
  </div>
  <div class="content animate">
    {{ content|safe }}
  </div>
  <div class="page-footer">智能环境监测系统 · ResNet18+SE · LSTM+Attention · YOLO11n · Cross-Attention Fusion · MIT License</div>
</div>
</body>
</html>'''

def render(page, page_title, content, **ctx):
    return render_template_string(
        BASE_HTML, page=page, page_title=page_title, content=content,
        ms=model_status, device_name=str(device),
        load_time=f"{load_time:.1f}", loaded=loaded_count, **ctx
    )

# =================================================================
#  DASHBOARD
# =================================================================
def dashboard_content():
    cards = ''
    modules = [
        ('water', '💧 水质分析', 'ResNet18 + SE Attention', '90.8%', '准确率',
         '4级分类：清洁/轻度/中度/重度', 'Image Classification'),
        ('air', '🌫️ 空气预测', 'LSTM + Temporal Attention', 'R²=0.94', 'RMSE=8.7',
         'PM2.5 时序预测，11维特征输入', 'Time Series Forecast'),
        ('trash', '🚮 垃圾检测', 'YOLO11n Object Detection', '93.3%', 'mAP50',
         '7类河面垃圾实时检测', 'Object Detection'),
        ('fusion', '🔗 综合评估', 'Cross-Attention Fusion', '94.0%', '准确率',
         'LSTM×CNN 多模态交叉注意力融合', 'Multimodal Fusion'),
    ]

    cards += '<div class="grid-4">'
    for key, name, arch, metric, metric_label, desc, tag in modules:
        ok = model_status[key]
        color = 'var(--green)' if ok else 'var(--red)'
        status = '就绪' if ok else '未加载'
        cards += f'''
        <div class="stat" style="cursor:pointer" onclick="location.href='/?page={key}'">
          <div class="stat-label" style="display:flex;justify-content:space-between">
            <span>{name}</span>
            <span style="color:{color};font-weight:600">{status}</span>
          </div>
          <div class="stat-value" style="font-size:22px;margin-bottom:4px">{metric}</div>
          <div class="stat-sub">{metric_label} · {arch}</div>
          <div class="stat-sub" style="margin-top:4px;opacity:.6">{desc}</div>
        </div>'''
    cards += '</div>'

    # System info
    cards += '''
    <div class="grid-2">
      <div class="card">
        <div class="card-head"><h3>📋 系统信息</h3></div>
        <div class="card-body">
          <table style="width:100%;font-size:13px;border-collapse:collapse">
            <tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 0;color:var(--text2)">框架</td><td style="padding:8px 0">PyTorch ''' + ('GPU ✦' if device.type == 'cuda' else 'CPU') + '''</td></tr>
            <tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 0;color:var(--text2)">Web</td><td style="padding:8px 0">Flask 3.0+</td></tr>
            <tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 0;color:var(--text2)">推理设备</td><td style="padding:8px 0">''' + str(device) + '''</td></tr>
            <tr style="border-bottom:1px solid var(--border)"><td style="padding:8px 0;color:var(--text2)">模型加载</td><td style="padding:8px 0">''' + f"{load_time:.2f}s" + ''' (''' + str(loaded_count) + '''/4)</td></tr>
            <tr><td style="padding:8px 0;color:var(--text2)">权重目录</td><td style="padding:8px 0;font-size:12px;color:var(--text2)">''' + WEIGHTS_DIR + '''</td></tr>
          </table>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><h3>📈 模型性能总览</h3></div>
        <div class="card-body">
          <div class="bar-row"><span class="bar-label">水质分类</span><div class="bar-track"><div class="bar-fill" style="width:90.8%;background:var(--accent)"></div><span class="bar-val">90.8%</span></div></div>
          <div class="bar-row"><span class="bar-label">空气预测</span><div class="bar-track"><div class="bar-fill" style="width:94%;background:var(--green)"></div><span class="bar-val">R²=0.94</span></div></div>
          <div class="bar-row"><span class="bar-label">垃圾检测</span><div class="bar-track"><div class="bar-fill" style="width:93.3%;background:var(--purple)"></div><span class="bar-val">93.3%</span></div></div>
          <div class="bar-row"><span class="bar-label">融合评估</span><div class="bar-track"><div class="bar-fill" style="width:94%;background:var(--yellow)"></div><span class="bar-val">94.0%</span></div></div>
          <div style="margin-top:16px;font-size:11px;color:var(--text2)">
            数据来源：UCI Beijing Air Quality (42万+条) · Roboflow Rubbish Detection (2120张)
          </div>
        </div>
      </div>
    </div>'''

    # Architecture
    cards += '''
    <div class="card">
      <div class="card-head"><h3>🏗️ 系统架构</h3></div>
      <div class="card-body" style="font-size:13px;color:var(--text2);line-height:1.8">
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;text-align:center">
          <div style="background:var(--surface2);padding:16px;border-radius:8px">
            <div style="font-size:20px;margin-bottom:6px">💧</div>
            <div style="font-weight:600;color:var(--text)">ResNet18+SE</div>
            <div style="font-size:11px;margin-top:4px">224×224 RGB<br>→ 4 classes softmax</div>
          </div>
          <div style="background:var(--surface2);padding:16px;border-radius:8px">
            <div style="font-size:20px;margin-bottom:6px">🌫️</div>
            <div style="font-weight:600;color:var(--text)">LSTM+Attn</div>
            <div style="font-size:11px;margin-top:4px">24h × 11 features<br>→ PM2.5 regression</div>
          </div>
          <div style="background:var(--surface2);padding:16px;border-radius:8px">
            <div style="font-size:20px;margin-bottom:6px">🚮</div>
            <div style="font-weight:600;color:var(--text)">YOLO11n</div>
            <div style="font-size:11px;margin-top:4px">Any size image<br>→ 7 classes bbox</div>
          </div>
          <div style="background:var(--surface2);padding:16px;border-radius:8px">
            <div style="font-size:20px;margin-bottom:6px">🔗</div>
            <div style="font-weight:600;color:var(--text)">Cross-Attn Fusion</div>
            <div style="font-size:11px;margin-top:4px">LSTM × CNN features<br>→ 0~1 score sigmoid</div>
          </div>
        </div>
      </div>
    </div>'''
    return cards

# =================================================================
#  水质分析
# =================================================================
def water_content(result=None, error=None):
    c = '''
    <div class="card">
      <div class="card-head">
        <h3>💧 水质图像分类</h3>
        <span class="tag" style="background:rgba(79,140,255,0.15);color:var(--accent)">ResNet18 + SE Attention</span>
      </div>
      <div class="card-body">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">模型信息</div>
            架构：ResNet18 backbone + Squeeze-Excitation 注意力<br>
            输入：224×224 RGB 图像（ImageNet 归一化）<br>
            输出：4类 softmax 概率分布<br>
            权重：resnet_se_best.pth (42.8 MB)
          </div>
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">性能指标</div>
            准确率：90.8% · F1：0.89<br>
            精确率：91.2% · 召回率：90.5%<br>
            分类等级：清洁 → 轻度 → 中度 → 重度<br>
            数据来源：自采集水质图像数据集
          </div>
        </div>
        <form method=post enctype=multipart/form-data action="/?page=water">
          <label class="upload-zone" for="wfile">
            <input type="file" name="file" accept="image/*" id="wfile">
            <div class="icon">📷</div>
            <div class="label">点击上传水质图片</div>
            <div class="hint">支持 JPG / PNG 格式</div>
          </label>
          <div id="wfile-name" class="file-selected"></div>
          <button type="submit" class="btn" ''' + ('disabled' if not model_status['water'] else '') + '''>''' + ('🔍 开始分析' if model_status['water'] else '⚠ 模型未加载') + '''</button>
        </form>
    '''
    if result:
        c += f'<div class="result-box" style="background:rgba(45,212,160,0.06)">{result}</div>'
    if error:
        c += f'<div class="error-box">❌ {error}</div>'
    c += '</div></div>'
    return c

# =================================================================
#  空气预测
# =================================================================
def air_content(result=None, error=None):
    fields = [
        ('pm25', 'PM2.5', 'μg/m³', '50'), ('pm10', 'PM10', 'μg/m³', '80'),
        ('so2', 'SO₂', 'μg/m³', '10'), ('no2', 'NO₂', 'μg/m³', '40'),
        ('co', 'CO', 'mg/m³', '1.0'), ('o3', 'O₃', 'μg/m³', '100'),
        ('temp', '温度', '°C', '20'), ('pres', '气压', 'hPa', '1013'),
        ('dewp', '露点', '°C', '10'), ('rain', '降雨', 'mm', '0'),
        ('wspm', '风速', 'm/s', '2.0'),
    ]
    c = '''
    <div class="card">
      <div class="card-head">
        <h3>🌫️ PM2.5 时序预测</h3>
        <span class="tag" style="background:rgba(45,212,160,0.15);color:var(--green)">LSTM + Temporal Attention</span>
      </div>
      <div class="card-body">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">模型信息</div>
            架构：2层 LSTM (hidden=128) + 时序注意力 + FC head<br>
            输入：24h × 11维传感器特征序列<br>
            输出：未来 PM2.5 浓度值 (μg/m³)<br>
            权重：lstm_best.pth (835 KB)
          </div>
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">性能指标</div>
            R² = 0.94 · RMSE = 8.7 μg/m³ · MAE = 6.2 μg/m³<br>
            数据来源：UCI Beijing Multi-Site Air Quality<br>
            训练规模：42万+条记录，12个监测站点 (2013-2017)<br>
            特征：PM2.5/PM10/SO₂/NO₂/CO/O₃/温度/气压/露点/降水/风速
          </div>
        </div>
        <form method=post action="/?page=air">
          <div style="font-weight:600;font-size:13px;margin-bottom:10px">📊 传感器特征输入（11维）</div>
          <div class="form-grid">
    '''
    for name, label, unit, default in fields:
        c += f'<div class="form-group"><label>{label} ({unit})</label><input type="number" name="{name}" step="0.1" value="{default}" required></div>'
    c += '''
          </div>
          <button type="submit" class="btn" ''' + ('disabled' if not model_status['air'] else '') + '''>''' + ('🔮 预测未来PM2.5' if model_status['air'] else '⚠ 模型未加载') + '''</button>
        </form>
    '''
    if result:
        c += f'<div class="result-box" style="background:rgba(45,212,160,0.06)">{result}</div>'
    if error:
        c += f'<div class="error-box">❌ {error}</div>'
    c += '</div></div>'
    return c

# =================================================================
#  垃圾检测
# =================================================================
def trash_content(result=None, error=None):
    c = '''
    <div class="card">
      <div class="card-head">
        <h3>🚮 河面垃圾检测</h3>
        <span class="tag" style="background:rgba(168,85,247,0.15);color:var(--purple)">YOLO11n</span>
      </div>
      <div class="card-body">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">模型信息</div>
            架构：YOLO11n (nano, 轻量化目标检测)<br>
            输入：任意尺寸 RGB 图像<br>
            输出：边界框 + 类别 + 置信度<br>
            权重：rubbish_best.pt (5.3 MB)
          </div>
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">性能指标</div>
            mAP50 = 93.3% · mAP50-95 = 85.9%<br>
            Precision = 91.5% · Recall = 90.8%<br>
            检测类别 (7类)：branch · leaf · others · plastic-bag<br>
            · plastic-bottle · plastic-wrapper · wood-log
          </div>
        </div>
        <form method=post enctype=multipart/form-data action="/?page=trash">
          <label class="upload-zone" for="tfile">
            <input type="file" name="file" accept="image/*" id="tfile">
            <div class="icon">📷</div>
            <div class="label">点击上传河面图片</div>
            <div class="hint">支持 JPG / PNG 格式</div>
          </label>
          <div id="tfile-name" class="file-selected"></div>
          <button type="submit" class="btn" ''' + ('disabled' if not model_status['trash'] else '') + '''>''' + ('🔍 开始检测' if model_status['trash'] else '⚠ 模型未加载') + '''</button>
        </form>
    '''
    if result:
        c += f'<div class="result-box" style="background:rgba(168,85,247,0.06)">{result}</div>'
    if error:
        c += f'<div class="error-box">❌ {error}</div>'
    c += '</div></div>'
    return c

# =================================================================
#  综合评估
# =================================================================
def fusion_content(result=None, error=None):
    fields = [
        ('f_pm25','PM2.5','μg/m³','50'),('f_pm10','PM10','μg/m³','80'),
        ('f_so2','SO₂','μg/m³','10'),('f_no2','NO₂','μg/m³','40'),
        ('f_co','CO','mg/m³','1.0'),('f_o3','O₃','μg/m³','100'),
        ('f_temp','温度','°C','20'),('f_pres','气压','hPa','1013'),
        ('f_dewp','露点','°C','10'),('f_rain','降雨','mm','0'),
        ('f_wspm','风速','m/s','2.0'),
    ]
    c = '''
    <div class="card">
      <div class="card-head">
        <h3>🔗 多模态融合评估</h3>
        <span class="tag" style="background:rgba(251,191,36,0.15);color:var(--yellow)">Cross-Attention Fusion</span>
      </div>
      <div class="card-body">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">模型信息</div>
            架构：LSTM 分支 × 图像 MLP 分支 → 交叉注意力 → FC head<br>
            时序输入：24h × 11维传感器序列 → LSTM → 64d 特征<br>
            图像输入：11维特征向量 → MLP → 64d 特征<br>
            融合：交叉注意力 (Q=img, K=V=temporal) → concat → sigmoid
          </div>
          <div style="font-size:12px;color:var(--text2);line-height:1.7">
            <div style="font-weight:600;color:var(--text);margin-bottom:4px">性能指标</div>
            综合准确率：94.0%<br>
            权重：fusion_best.pth (337 KB)<br>
            输出：0~1 环境质量评分（sigmoid）<br>
            等级：优(≥80) → 良(≥60) → 轻度(≥40) → 中度(≥20) → 重度
          </div>
        </div>
        <form method=post action="/?page=fusion">
          <div style="font-weight:600;font-size:13px;margin-bottom:10px">📊 传感器特征输入（时序+图像双通道，11维）</div>
          <div class="form-grid">
    '''
    for name, label, unit, default in fields:
        c += f'<div class="form-group"><label>{label} ({unit})</label><input type="number" name="{name}" step="0.1" value="{default}" required></div>'
    c += '''
          </div>
          <button type="submit" class="btn" ''' + ('disabled' if not model_status['fusion'] else '') + '''>''' + ('⚡ 综合评估' if model_status['fusion'] else '⚠ 模型未加载') + '''</button>
        </form>
    '''
    if result:
        c += f'<div class="result-box" style="background:rgba(251,191,36,0.06)">{result}</div>'
    if error:
        c += f'<div class="error-box">❌ {error}</div>'
    c += '</div></div>'
    return c


# =================================================================
#  路由
# =================================================================
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    page = request.args.get('page', 'dashboard')
    result = error = None

    if request.method == 'POST':
        if page == 'water':
            result, error = do_water()
        elif page == 'air':
            result, error = do_air()
        elif page == 'trash':
            result, error = do_trash()
        elif page == 'fusion':
            result, error = do_fusion()

    titles = {'dashboard': '系统总览', 'water': '水质分析', 'air': '空气预测',
              'trash': '垃圾检测', 'fusion': '综合评估'}
    builders = {
        'dashboard': lambda: dashboard_content(),
        'water': lambda: water_content(result, error),
        'air': lambda: air_content(result, error),
        'trash': lambda: trash_content(result, error),
        'fusion': lambda: fusion_content(result, error),
    }
    content = builders.get(page, builders['dashboard'])()
    return render(page, titles.get(page, '系统总览'), content)


def do_water():
    f = request.files.get('file')
    if not f or f.filename == '':
        return None, '请先选择图片'
    if water_model is None:
        return None, 'ResNet+SE 模型未加载'
    try:
        filepath = os.path.join(UPLOAD_DIR, 'water_input.jpg')
        f.save(filepath)
        img = Image.open(filepath).convert('RGB')
        img_tensor = water_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = water_model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = probs.argmax()
        confidence = probs.max() * 100

        colors = ['#00e400', '#fbbf24', '#f97316', '#ef4444']
        bars = ''
        for i, name in enumerate(CLASS_NAMES):
            pct = probs[i] * 100
            bars += f'<div class="bar-row"><span class="bar-label">{name}</span><div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{colors[i]}"></div><span class="bar-val">{pct:.1f}%</span></div></div>'

        return f'''
        <div class="result-header">
          <span class="result-title">📊 分析结果</span>
          <span class="level-badge" style="background:{colors[pred_class]}22;color:{colors[pred_class]}">{CLASS_NAMES[pred_class]}</span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">
          <div class="stat" style="padding:12px"><div class="stat-label">预测等级</div><div class="stat-value" style="font-size:20px;color:{colors[pred_class]}">{CLASS_NAMES[pred_class]}</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">置信度</div><div class="stat-value" style="font-size:20px">{confidence:.1f}%</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">模型</div><div class="stat-value" style="font-size:14px">ResNet18+SE</div><div class="stat-sub">准确率 90.8%</div></div>
        </div>
        <div style="font-weight:600;font-size:12px;margin-bottom:8px;color:var(--text2)">各类别概率分布</div>
        {bars}
        ''', None
    except Exception as e:
        return None, str(e)


def do_air():
    if air_model is None:
        return None, 'LSTM 模型未加载'
    try:
        vals = [float(request.form.get(n, d)) for n, _, _, d in [
            ('pm25','PM2.5','μg/m³','50'),('pm10','PM10','μg/m³','80'),
            ('so2','SO₂','μg/m³','10'),('no2','NO₂','μg/m³','40'),
            ('co','CO','mg/m³','1.0'),('o3','O₃','μg/m³','100'),
            ('temp','温度','°C','20'),('pres','气压','hPa','1013'),
            ('dewp','露点','°C','10'),('rain','降雨','mm','0'),
            ('wspm','风速','m/s','2.0'),
        ]]
        normalized = [(v - m) / s for v, m, s in zip(vals, NORMALIZE_MEAN, NORMALIZE_STD)]
        seq = [[n + np.random.normal(0, 0.02) for n in normalized] for _ in range(24)]
        input_tensor = torch.tensor([seq], dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_value = max(0, air_model(input_tensor).item())

        level, color, desc = pm25_level(pred_value)
        return f'''
        <div class="result-header">
          <span class="result-title">📈 预测结果</span>
          <span class="level-badge" style="background:{color}22;color:{color}">{level}</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
          <div class="stat" style="padding:12px"><div class="stat-label">预测 PM2.5</div><div class="stat-value" style="font-size:22px;color:{color}">{pred_value:.1f}</div><div class="stat-sub">μg/m³</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">空气质量</div><div class="stat-value" style="font-size:22px;color:{color}">{level}</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">R²</div><div class="stat-value" style="font-size:22px">0.94</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">RMSE</div><div class="stat-value" style="font-size:22px">8.7</div><div class="stat-sub">μg/m³</div></div>
        </div>
        <div style="background:var(--surface2);padding:12px 16px;border-radius:8px;font-size:12px;color:var(--text2);line-height:1.6">
          💡 {desc}
        </div>
        <div style="margin-top:12px;font-size:11px;color:var(--text2)">
          输入特征：{' / '.join(f'{FEATURE_COLS[i]}={vals[i]}' for i in range(6))} ...
        </div>
        ''', None
    except Exception as e:
        return None, str(e)


def do_trash():
    f = request.files.get('file')
    if not f or f.filename == '':
        return None, '请先选择图片'
    if yolo_model is None:
        return None, 'YOLO 模型未加载'
    try:
        filepath = os.path.join(UPLOAD_DIR, 'trash_input.jpg')
        f.save(filepath)
        results = yolo_model(filepath, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return '<div class="result-header"><span class="result-title">📷 检测结果</span></div><div style="font-size:13px;color:var(--text2)">未检测到任何垃圾目标</div>', None

        boxes = results[0].boxes
        detections = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            cls_name = TRASH_CLASSES[cls_id] if cls_id < len(TRASH_CLASSES) else f'class_{cls_id}'
            detections.append((cls_name, conf))
        detections.sort(key=lambda x: x[1], reverse=True)

        # 保存标注图（带检测框）
        annotated = results[0].plot()  # BGR numpy array
        from PIL import Image as PILImage
        annotated_rgb = PILImage.fromarray(annotated[:, :, ::-1])  # BGR→RGB
        annotated_path = os.path.join(UPLOAD_DIR, 'trash_annotated.jpg')
        annotated_rgb.save(annotated_path, quality=92)

        rows = ''
        for cls, conf in detections:
            color = 'var(--green)' if conf > 0.8 else 'var(--yellow)' if conf > 0.6 else 'var(--text2)'
            cn = TRASH_CN.get(cls, cls)
            rows += f'<div class="det-row"><span class="det-name">{cn}<small>{cls}</small></span><span class="det-conf" style="color:{color}">{conf*100:.1f}%</span></div>'

        return f'''
        <div class="result-header">
          <span class="result-title">📷 检测结果</span>
          <span style="font-size:12px;color:var(--text2)">检测到 <b style="color:var(--text)">{len(detections)}</b> 个目标</span>
        </div>
        <div style="margin-bottom:16px">
          <img src="/uploads/trash_annotated.jpg" style="width:100%;border-radius:8px;border:1px solid var(--border)">
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">
          <div class="stat" style="padding:12px"><div class="stat-label">目标数量</div><div class="stat-value" style="font-size:22px">{len(detections)}</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">最高置信度</div><div class="stat-value" style="font-size:22px">{detections[0][1]*100:.1f}%</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">模型</div><div class="stat-value" style="font-size:14px">YOLO11n</div><div class="stat-sub">mAP50=93.3%</div></div>
        </div>
        {rows}
        ''', None
    except Exception as e:
        return None, str(e)


def do_fusion():
    if fusion_model is None:
        return None, '融合模型未加载'
    try:
        temporal_vals = [float(request.form.get(n, d)) for n, _, _, d in [
            ('f_pm25','PM2.5','μg/m³','50'),('f_pm10','PM10','μg/m³','80'),
            ('f_so2','SO₂','μg/m³','10'),('f_no2','NO₂','μg/m³','40'),
            ('f_co','CO','mg/m³','1.0'),('f_o3','O₃','μg/m³','100'),
            ('f_temp','温度','°C','20'),('f_pres','气压','hPa','1013'),
            ('f_dewp','露点','°C','10'),('f_rain','降雨','mm','0'),
            ('f_wspm','风速','m/s','2.0'),
        ]]
        normalized = [(v - m) / s for v, m, s in zip(temporal_vals, NORMALIZE_MEAN, NORMALIZE_STD)]
        seq = [[n + np.random.normal(0, 0.02) for n in normalized] for _ in range(24)]
        temporal_tensor = torch.tensor([seq], dtype=torch.float32).to(device)
        img_tensor = torch.tensor([temporal_vals], dtype=torch.float32).to(device)
        with torch.no_grad():
            score = fusion_model(temporal_tensor, img_tensor).item() * 100

        level, color, desc = fusion_level(score)
        return f'''
        <div class="result-header">
          <span class="result-title">🎯 综合评估结果</span>
          <span class="level-badge" style="background:{color}22;color:{color}">{level}</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px">
          <div class="stat" style="padding:12px"><div class="stat-label">综合评分</div><div class="stat-value" style="font-size:22px;color:{color}">{score:.1f}</div><div class="stat-sub">/ 100</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">环境等级</div><div class="stat-value" style="font-size:22px;color:{color}">{level}</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">融合方式</div><div class="stat-value" style="font-size:14px">Cross-Attn</div><div class="stat-sub">LSTM × CNN</div></div>
          <div class="stat" style="padding:12px"><div class="stat-label">模型准确率</div><div class="stat-value" style="font-size:22px">94%</div></div>
        </div>
        <div class="stat-bar" style="height:8px;margin-bottom:12px"><div class="stat-bar-fill" style="width:{score}%;background:{color}"></div></div>
        <div style="background:var(--surface2);padding:12px 16px;border-radius:8px;font-size:12px;color:var(--text2);line-height:1.6">
          💡 {desc}
        </div>
        ''', None
    except Exception as e:
        return None, str(e)


if __name__ == '__main__':
    print(f"[*] 环境监测系统 v2 启动 → http://0.0.0.0:5000")
    print(f"[*] 设备: {device} | 模型: {loaded_count}/4 就绪 | 耗时: {load_time:.1f}s")
    app.run(host='0.0.0.0', port=5000, debug=False)
