# -*- mode: python ; coding: utf-8 -*-
"""
LWF.spec — Lite-Whisper-Faster v2.0 PyInstaller 打包配置

使用方式：
    pyinstaller LWF.spec

输出结构 (dist/LWF/):
    LWF/
    ├── LWF.exe
    ├── bin/               # ffmpeg + CUDA dll
    ├── models/            # Whisper 模型
    ├── temp/              # 运行时临时目录
    └── _internal/         # PyInstaller 依赖

打包策略：
    - onedir 模式：方便替换 bin/models
    - 排除 PyTorch / TensorFlow 等巨型库
    - 收集 ctranslate2 / faster_whisper / av / customtkinter 核心依赖
"""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# ═══════════════════════════════════════════════════════════════════════════
# 项目路径
# ═══════════════════════════════════════════════════════════════════════════

SPEC_DIR = os.path.abspath(os.path.dirname(SPECPATH))
PROJECT_DIR = SPEC_DIR

# ═══════════════════════════════════════════════════════════════════════════
# 排除巨型库
# ═══════════════════════════════════════════════════════════════════════════

EXCLUDES = [
    # PyTorch
    "torch", "torchvision", "torchaudio", "torch._C", "torch.utils",
    # TensorFlow
    "tensorflow", "tensorflow_gpu", "tf2onnx", "tensorboard",
    "tensorboard_data_server", "tensorboard_plugin_wit", "keras",
    # JAX
    "jax", "jaxlib", "flax", "optax",
    # 大型 ML 库
    "transformers", "datasets", "accelerate", "diffusers", "safetensors",
    "onnx", "scipy", "scikit-learn", "sklearn",
    "pandas", "matplotlib", "plotly", "seaborn", "sympy",
    # NVIDIA Python 包
    "nvidia", "nvidia_cublas_cu12", "nvidia_cudnn_cu12",
    "nvidia_cuda_runtime_cu12", "nvidia_cuda_nvrtc_cu12",
    "nvidia_cufft_cu12", "nvidia_curand_cu12", "nvidia_cusolver_cu12",
    "nvidia_cusparse_cu12", "nvidia_nccl_cu12", "nvidia_nvjitlink_cu12",
    "triton",
    # 开发工具
    "pytest", "unittest", "IPython", "notebook", "jupyter",
    "jupyter_core", "jupyterlab", "nbformat", "nbconvert",
    "black", "isort", "pylint", "mypy", "setuptools", "pip", "wheel",
]

# ═══════════════════════════════════════════════════════════════════════════
# 依赖收集
# ═══════════════════════════════════════════════════════════════════════════

ct2_datas, ct2_binaries, ct2_hiddenimports = collect_all("ctranslate2")
fw_datas, fw_binaries, fw_hiddenimports = collect_all("faster_whisper")
av_datas, av_binaries, av_hiddenimports = collect_all("av")
tk_binaries = collect_dynamic_libs("tokenizers")
tk_hiddenimports = collect_submodules("tokenizers")
hf_datas = collect_data_files("huggingface_hub")
hf_hiddenimports = collect_submodules("huggingface_hub")
ctk_datas, ctk_binaries, ctk_hiddenimports = collect_all("customtkinter")

# onnxruntime (VAD)
try:
    ort_datas, ort_binaries, ort_hiddenimports = collect_all("onnxruntime")
except Exception:
    ort_datas, ort_binaries, ort_hiddenimports = [], [], []

# ═══════════════════════════════════════════════════════════════════════════
# 合并
# ═══════════════════════════════════════════════════════════════════════════

all_datas = ct2_datas + fw_datas + av_datas + hf_datas + ctk_datas + ort_datas
all_binaries = ct2_binaries + fw_binaries + av_binaries + tk_binaries + ctk_binaries + ort_binaries
all_hiddenimports = list(set(
    ct2_hiddenimports + fw_hiddenimports + av_hiddenimports
    + tk_hiddenimports + hf_hiddenimports + ctk_hiddenimports
    + ort_hiddenimports
    + [
        "ctranslate2", "faster_whisper", "av", "av.audio", "av.audio.frame",
        "huggingface_hub", "tokenizers", "customtkinter", "onnxruntime",
    ]
))

# ═══════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════

a = Analysis(
    [os.path.join(PROJECT_DIR, "main_ui.py")],
    pathex=[PROJECT_DIR],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts,
    exclude_binaries=True,
    name="LWF",
    icon=None,   # TODO: icon=os.path.join(PROJECT_DIR, "assets", "icon.ico")
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    uac_admin=False,
    uac_uiaccess=False,
    version=None,  # TODO: version="version_info.txt"
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name="LWF",
)
