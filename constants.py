"""
constants.py — LWF 全局常量定义

集中管理版本号、模型信息、支持格式等常量。
所有模块统一从这里引用，避免硬编码散落各处。
"""

# ═══════════════════════════════════════════════════════════════════════════
# 应用信息
# ═══════════════════════════════════════════════════════════════════════════

APP_NAME = "Lite-Whisper-Faster"
APP_VERSION = "2.3.0"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"

# ═══════════════════════════════════════════════════════════════════════════
# Whisper 模型定义
# ═══════════════════════════════════════════════════════════════════════════
#
# faster-whisper 使用 CTranslate2 格式的模型，官方托管在 HuggingFace。
# 每个模型是一个文件夹，包含 model.bin + config.json + tokenizer.json 等。

WHISPER_MODELS = {
    "tiny": {
        "repo": "Systran/faster-whisper-tiny",
        "size_mb": 75,
        "desc": "极速模式，适合快速测试",
        "params": "39M",
        "vram_gb": 1.0,
    },
    "base": {
        "repo": "Systran/faster-whisper-base",
        "size_mb": 145,
        "desc": "日常推荐，速度与精度均衡",
        "params": "74M",
        "vram_gb": 1.0,
    },
    "small": {
        "repo": "Systran/faster-whisper-small",
        "size_mb": 484,
        "desc": "较高精度，适合正式场景",
        "params": "244M",
        "vram_gb": 2.0,
    },
    "medium": {
        "repo": "Systran/faster-whisper-medium",
        "size_mb": 1500,
        "desc": "高精度，需 4GB+ 内存",
        "params": "769M",
        "vram_gb": 4.0,
    },
    "large-v3": {
        "repo": "Systran/faster-whisper-large-v3",
        "size_mb": 3000,
        "desc": "最高精度，需 6GB+ 显存",
        "params": "1550M",
        "vram_gb": 6.0,
    },
}

DEFAULT_MODEL = "base"

# ═══════════════════════════════════════════════════════════════════════════
# 语言选项
# ═══════════════════════════════════════════════════════════════════════════

LANGUAGE_OPTIONS = [
    ("auto", "自动检测"),
    ("zh", "中文"),
    ("en", "英语"),
    ("ja", "日语"),
    ("ko", "韩语"),
    ("fr", "法语"),
    ("de", "德语"),
    ("es", "西班牙语"),
    ("ru", "俄语"),
    ("pt", "葡萄牙语"),
    ("ar", "阿拉伯语"),
]

DEFAULT_LANGUAGE = "auto"

# ═══════════════════════════════════════════════════════════════════════════
# 支持的文件格式
# ═══════════════════════════════════════════════════════════════════════════

SUPPORTED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".ts"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aac", ".opus"}
SUPPORTED_EXTS = SUPPORTED_VIDEO_EXTS | SUPPORTED_AUDIO_EXTS

# tkinter filedialog 格式字符串
FILEDIALOG_TYPES = [
    ("音视频文件", " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTS))),
    ("视频文件", " ".join(f"*{ext}" for ext in sorted(SUPPORTED_VIDEO_EXTS))),
    ("音频文件", " ".join(f"*{ext}" for ext in sorted(SUPPORTED_AUDIO_EXTS))),
    ("所有文件", "*.*"),
]

# ═══════════════════════════════════════════════════════════════════════════
# CTranslate2 模型验证
# ═══════════════════════════════════════════════════════════════════════════
# 一个有效的 CTranslate2 模型文件夹至少包含以下文件

MODEL_REQUIRED_FILES = ["model.bin", "config.json", "tokenizer.json"]

# ═══════════════════════════════════════════════════════════════════════════
# 默认配置值
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "version": APP_VERSION,
    "default_model": DEFAULT_MODEL,
    "default_language": DEFAULT_LANGUAGE,
    "prefer_gpu": True,
    "hf_mirror": "",              # 空 = 官方源；"https://hf-mirror.com" = 国内镜像
    "beam_size": 5,
    "vad_filter": True,
    "vad_min_silence_ms": 500,
    "theme": "dark",              # dark / light
    "last_output_dir": "",
    "max_concurrent": 1,          # 预留：批量处理并发数
    # ── 翻译模块 ──
    "translate_provider": "DeepSeek",
    "translate_api_base": "https://api.deepseek.com/v1",
    "translate_api_key": "",
    "translate_model": "deepseek-chat",
    "translate_target_lang": "简体中文",
}

# ═══════════════════════════════════════════════════════════════════════════
# GPU 相关
# ═══════════════════════════════════════════════════════════════════════════

# bin/ 下需要检测的 CUDA DLL 关键字
GPU_DLL_PATTERNS = ["cudnn", "cublas", "cublaslt", "cudart"]

# 必需的 CUDA DLL 文件（CUDA 12 + cuDNN 9 组合）
REQUIRED_GPU_DLLS = [
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
]
