"""
utils_env.py — LWF 环境与路径初始化模块

职责：
  1. 确定应用根目录（兼容开发环境与 PyInstaller 打包后的环境）
  2. 将 /bin 目录注入系统 PATH 最前端，确保优先调用自带的 ffmpeg / dll
  3. 确保 /models、/temp 等目录存在
  4. 提供 ffmpeg 可用性检测
  5. 提供 GPU DLL 检测
  6. 提供临时文件清理功能

使用方式：
  在入口文件的最顶部调用：
      from utils_env import init_environment
      env = init_environment()
"""

import os
import sys
import shutil
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from constants import GPU_DLL_PATTERNS

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logger = logging.getLogger("LWF")


# ---------------------------------------------------------------------------
# 数据类：环境信息
# ---------------------------------------------------------------------------
@dataclass
class EnvInfo:
    """保存初始化后的环境路径信息，供其他模块统一引用。"""
    root_dir: Path          # 应用根目录
    bin_dir: Path           # /bin  — ffmpeg、dll 所在目录
    models_dir: Path        # /models — whisper 模型存放目录
    temp_dir: Path          # /temp — ffmpeg 临时 wav 文件
    config_path: Path       # config.json 路径
    ffmpeg_path: Optional[Path] = None   # ffmpeg.exe 的完整路径
    ffprobe_path: Optional[Path] = None  # ffprobe.exe 的完整路径
    gpu_available: bool = False           # 是否检测到 CUDA dll
    gpu_dlls_found: list = field(default_factory=list)  # 找到的 GPU dll 列表


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def _get_app_root() -> Path:
    """
    确定应用根目录。
    - PyInstaller 打包后：取 exe 所在的目录
    - 开发环境：取本文件所在目录
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent.resolve()
    else:
        return Path(__file__).parent.resolve()


def _inject_bin_to_path(bin_dir: Path) -> None:
    """
    将 bin 目录插入到系统 PATH 的最前端。
    两个关键目的：
      1. subprocess 调用 ffmpeg 时优先找到我们自带的版本
      2. Windows 加载 dll 时优先搜索 bin 目录（cudnn、cublas 等）
    """
    bin_str = str(bin_dir)
    current_path = os.environ.get("PATH", "")

    path_entries = current_path.split(os.pathsep)
    if bin_str.lower() not in [p.lower() for p in path_entries]:
        os.environ["PATH"] = bin_str + os.pathsep + current_path
        logger.info(f"[ENV] 已将 bin 目录注入 PATH 最前端: {bin_str}")
    else:
        logger.info(f"[ENV] bin 目录已在 PATH 中，跳过注入: {bin_str}")

    # Windows：同时设置 DLL 搜索路径
    if sys.platform == "win32":
        try:
            os.add_dll_directory(bin_str)
            logger.info(f"[ENV] 已通过 os.add_dll_directory 注册 DLL 搜索路径")
        except (OSError, AttributeError) as e:
            logger.warning(f"[ENV] os.add_dll_directory 不可用，依赖 PATH 兜底: {e}")


def _ensure_directories(dirs: dict[str, Path]) -> None:
    """确保所有必需的目录存在。"""
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[ENV] 目录就绪: {name} -> {path}")


def _detect_ffmpeg(bin_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """
    检测 ffmpeg 和 ffprobe 是否可用。
    优先检查 bin 目录下的文件，再 fallback 到系统 PATH。
    """
    ffmpeg_path = None
    ffprobe_path = None

    # 1. 优先检查 bin 目录
    local_ffmpeg = bin_dir / "ffmpeg.exe"
    local_ffprobe = bin_dir / "ffprobe.exe"

    if local_ffmpeg.is_file():
        ffmpeg_path = local_ffmpeg
        logger.info(f"[ENV] 找到本地 ffmpeg: {ffmpeg_path}")
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            ffmpeg_path = Path(system_ffmpeg)
            logger.info(f"[ENV] 使用系统 ffmpeg: {ffmpeg_path}")
        else:
            logger.warning("[ENV] ⚠ 未找到 ffmpeg！音视频文件将无法处理。")

    if local_ffprobe.is_file():
        ffprobe_path = local_ffprobe
        logger.info(f"[ENV] 找到本地 ffprobe: {ffprobe_path}")
    else:
        system_ffprobe = shutil.which("ffprobe")
        if system_ffprobe:
            ffprobe_path = Path(system_ffprobe)
            logger.info(f"[ENV] 使用系统 ffprobe: {ffprobe_path}")
        else:
            logger.warning("[ENV] ⚠ 未找到 ffprobe！媒体信息探测将不可用。")

    # 验证 ffmpeg 能否正常执行
    if ffmpeg_path:
        try:
            result = subprocess.run(
                [str(ffmpeg_path), "-version"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            version_line = result.stdout.split("\n")[0] if result.stdout else "未知版本"
            logger.info(f"[ENV] ffmpeg 版本: {version_line}")
        except Exception as e:
            logger.error(f"[ENV] ffmpeg 执行测试失败: {e}")
            ffmpeg_path = None

    return ffmpeg_path, ffprobe_path


def _detect_gpu_dlls(bin_dir: Path) -> tuple[bool, list[str]]:
    """
    检测 bin 目录下是否存在 CUDA 相关的 dll 文件。
    """
    found_dlls = []

    if bin_dir.is_dir():
        for f in bin_dir.iterdir():
            if f.suffix.lower() == ".dll":
                name_lower = f.stem.lower()
                if any(pattern in name_lower for pattern in GPU_DLL_PATTERNS):
                    found_dlls.append(f.name)

    if found_dlls:
        logger.info(f"[ENV] 检测到 {len(found_dlls)} 个 GPU 加速 DLL: {found_dlls}")
        return True, found_dlls
    else:
        logger.info("[ENV] 未检测到 GPU DLL，将使用 CPU 模式。")
        return False, []


def clean_temp(temp_dir: Path) -> int:
    """清空 temp 目录中的临时文件。返回被清理的文件数量。"""
    count = 0
    if not temp_dir.is_dir():
        return count

    for item in temp_dir.iterdir():
        try:
            if item.is_file() and item.name != ".gitkeep":
                item.unlink()
                count += 1
            elif item.is_dir():
                shutil.rmtree(item)
                count += 1
        except Exception as e:
            logger.warning(f"[ENV] 清理临时文件失败: {item} -> {e}")

    if count > 0:
        logger.info(f"[ENV] 已清理 {count} 个临时文件/目录")
    return count


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def init_environment(clean_temp_on_start: bool = True) -> EnvInfo:
    """
    一键初始化 LWF 运行环境。

    执行顺序：
      1. 确定应用根目录
      2. 注入 /bin 到 PATH（最高优先级）
      3. 创建缺失的目录
      4. 检测 ffmpeg
      5. 检测 GPU dll
      6. （可选）清空 temp 目录

    返回:
        EnvInfo 数据类实例
    """
    # 配置基础日志
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.info("=" * 50)
    logger.info("[ENV] LWF v2.0 环境初始化开始...")
    logger.info("=" * 50)

    # 1. 确定根目录
    root = _get_app_root()
    logger.info(f"[ENV] 应用根目录: {root}")

    # 2. 计算各子目录路径
    bin_dir = root / "bin"
    models_dir = root / "models"
    temp_dir = root / "temp"
    config_path = root / "config.json"

    # 3. 注入 bin 到 PATH
    if bin_dir.is_dir():
        _inject_bin_to_path(bin_dir)
    else:
        logger.warning(f"[ENV] ⚠ bin 目录不存在: {bin_dir}")

    # 4. 确保目录存在
    _ensure_directories({
        "models": models_dir,
        "temp": temp_dir,
    })

    # 5. 检测 ffmpeg
    ffmpeg_path, ffprobe_path = _detect_ffmpeg(bin_dir)

    # 6. 检测 GPU dll
    gpu_available, gpu_dlls = _detect_gpu_dlls(bin_dir)

    # 7. 清理临时目录
    if clean_temp_on_start:
        clean_temp(temp_dir)

    # 组装环境信息
    env = EnvInfo(
        root_dir=root,
        bin_dir=bin_dir,
        models_dir=models_dir,
        temp_dir=temp_dir,
        config_path=config_path,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        gpu_available=gpu_available,
        gpu_dlls_found=gpu_dlls,
    )

    logger.info("=" * 50)
    logger.info("[ENV] 环境初始化完成")
    logger.info(f"  根目录:     {env.root_dir}")
    logger.info(f"  ffmpeg:     {env.ffmpeg_path or '未找到'}")
    logger.info(f"  GPU 加速:   {'可用' if env.gpu_available else '不可用 (CPU 模式)'}")
    logger.info("=" * 50)

    return env


# ---------------------------------------------------------------------------
# 直接执行时用于自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("[*] LWF v2.0 环境自检\n")
    env = init_environment(clean_temp_on_start=False)
    print(f"\n[DIR] 根目录:   {env.root_dir}")
    print(f"[DIR] bin:      {env.bin_dir} {'[OK]' if env.bin_dir.is_dir() else '[MISSING]'}")
    print(f"[DIR] models:   {env.models_dir} {'[OK]' if env.models_dir.is_dir() else '[MISSING]'}")
    print(f"[DIR] temp:     {env.temp_dir} {'[OK]' if env.temp_dir.is_dir() else '[MISSING]'}")
    print(f"[CFG] config:   {env.config_path} {'[OK]' if env.config_path.is_file() else '[待创建]'}")
    print(f"[BIN] ffmpeg:   {env.ffmpeg_path or '[未找到]'}")
    print(f"[BIN] ffprobe:  {env.ffprobe_path or '[未找到]'}")
    print(f"[GPU] 加速:     {'可用 [OK]' if env.gpu_available else 'CPU 模式'}")
    if env.gpu_dlls_found:
        for dll in env.gpu_dlls_found:
            print(f"       - {dll}")
