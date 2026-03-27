# Lite-Whisper-Faster (LWF) v2.0

> 🎙 极简、免安装、开箱即用的本地视频/音频转文字桌面工具

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) 引擎，提供可视化操作界面。

## ✨ 特性

- **便携免安装** — 解压即用，无需安装 Python / CUDA / FFmpeg
- **可视化界面** — 基于 CustomTkinter 的现代深色 UI
- **模型管理** — 内置模型下载器，支持 HuggingFace 镜像加速
- **GPU 加速** — 自动检测 CUDA，无 GPU 自动回退 CPU
- **流式转录** — 逐句实时显示，不用等到最后
- **多格式导出** — 纯文本 TXT / SRT 字幕 / 剪贴板复制

## 📦 目录结构

```
LWF/
├── LWF.exe                 # 主程序
├── bin/                    # FFmpeg + CUDA DLL
│   ├── ffmpeg.exe
│   ├── ffprobe.exe
│   └── *.dll (GPU 可选)
├── models/                 # Whisper 模型
├── temp/                   # 运行时临时文件
├── config.json             # 用户配置
└── _internal/              # 程序依赖
```

## 🚀 快速开始

1. 解压 `LWF-v2.0-Portable.zip`
2. 双击 `LWF.exe` 启动
3. 首次运行会引导你下载模型（推荐 `base`，约 145 MB）
4. 选择音视频文件 → 开始转录

## 🔧 开发环境

```bash
# 安装依赖
pip install -r requirements.txt

# 运行（开发模式）
python main_ui.py

# 打包
build.bat
```

### 依赖

| 组件 | 用途 |
|------|------|
| faster-whisper | Whisper 语音识别引擎 |
| ctranslate2 | 高性能推理引擎 |
| customtkinter | 现代化 UI 框架 |
| huggingface_hub | 模型下载 |
| onnxruntime | VAD 语音活动检测 |
| av (PyAV) | 音频解码 |

## 📋 支持的模型

| 模型 | 大小 | 参数 | 推荐显存 | 说明 |
|------|------|------|----------|------|
| tiny | 75 MB | 39M | 1 GB | 极速测试 |
| base | 145 MB | 74M | 1 GB | **日常推荐** |
| small | 484 MB | 244M | 2 GB | 较高精度 |
| medium | 1.5 GB | 769M | 4 GB | 高精度 |
| large-v3 | 3 GB | 1550M | 6 GB | 最高精度 |

## 🌐 国内加速

在「设置」标签页中，将 HuggingFace 镜像地址设为：

```
https://hf-mirror.com
```

## 📄 许可

MIT License
