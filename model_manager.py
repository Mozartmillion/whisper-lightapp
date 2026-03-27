"""
model_manager.py — LWF 模型管理系统

负责 Whisper 模型的扫描、验证、下载和删除。
是「模型管理」Tab 的后端逻辑。

核心功能：
  1. scan_local_models()   — 扫描 models/ 目录下已安装的有效模型
  2. download_model()      — 从 HuggingFace 下载指定模型（支持镜像）
  3. delete_model()        — 删除本地已下载的模型
  4. get_model_status()    — 获取所有模型的安装状态

使用方式：
    mm = ModelManager(models_dir=env.models_dir, hf_mirror="")
    installed = mm.scan_local_models()
    mm.download_model("base", progress_callback=lambda p: print(p))
"""

import os
import shutil
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

from constants import WHISPER_MODELS, MODEL_REQUIRED_FILES

logger = logging.getLogger("LWF")


# ═══════════════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelInfo:
    """单个模型的状态信息"""
    name: str                     # 模型名（如 "base"）
    repo: str                     # HuggingFace 仓库名
    size_mb: int                  # 预估大小 (MB)
    desc: str                     # 描述
    params: str                   # 参数量
    vram_gb: float                # 推荐显存
    installed: bool = False       # 是否已安装
    local_path: Optional[Path] = None  # 本地路径（已安装时）
    local_size_mb: float = 0.0    # 实际占用大小 (MB)


@dataclass
class DownloadProgress:
    """下载进度信息"""
    model_name: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bps: float = 0.0       # 字节/秒
    file_name: str = ""          # 当前下载的文件名
    file_index: int = 0          # 当前文件序号
    file_total: int = 0          # 总文件数
    status: str = "downloading"  # downloading / completed / error / cancelled

    @property
    def progress(self) -> float:
        """0.0 ~ 1.0"""
        if self.total_bytes <= 0:
            return 0.0
        return min(self.downloaded_bytes / self.total_bytes, 1.0)

    @property
    def speed_mb(self) -> float:
        """MB/s"""
        return self.speed_bps / 1024 / 1024

    @property
    def downloaded_mb(self) -> float:
        return self.downloaded_bytes / 1024 / 1024

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1024 / 1024


# ═══════════════════════════════════════════════════════════════════════════
# 模型管理器
# ═══════════════════════════════════════════════════════════════════════════

class ModelManager:
    """
    Whisper 模型的生命周期管理。

    参数:
        models_dir:  模型存放目录（如 app_root/models/）
        hf_mirror:   HuggingFace 镜像地址，空字符串 = 使用官方源
    """

    def __init__(self, models_dir: Path, hf_mirror: str = ""):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_mirror = hf_mirror
        self._cancel_event = threading.Event()

    # ── 模型扫描 ──────────────────────────────────────────────────────

    def scan_local_models(self) -> dict[str, ModelInfo]:
        """
        扫描 models/ 下的所有模型，返回完整状态字典。
        会对照 WHISPER_MODELS 的定义和实际目录交叉检查。
        """
        result = {}

        for name, meta in WHISPER_MODELS.items():
            info = ModelInfo(
                name=name,
                repo=meta["repo"],
                size_mb=meta["size_mb"],
                desc=meta["desc"],
                params=meta["params"],
                vram_gb=meta["vram_gb"],
            )

            # 检查可能的本地目录名称
            # HuggingFace 下载的模型目录名格式：models--Systran--faster-whisper-base 的 snapshot
            # 或用户手动放入的文件夹名，如 "faster-whisper-base" 或直接 "base"
            local_path = self._find_model_path(name)
            if local_path:
                info.installed = True
                info.local_path = local_path
                info.local_size_mb = self._get_dir_size_mb(local_path)

            result[name] = info

        installed_count = sum(1 for m in result.values() if m.installed)
        logger.info(
            f"[MODEL] 扫描完成: {installed_count}/{len(result)} 个模型已安装"
        )
        return result

    def get_installed_model_names(self) -> list[str]:
        """快速返回已安装的模型名列表"""
        models = self.scan_local_models()
        return [name for name, info in models.items() if info.installed]

    # ── 模型下载 ──────────────────────────────────────────────────────

    def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> Path:
        """
        从 HuggingFace 下载指定模型到 models/ 目录。

        参数:
            model_name:        模型名（如 "base"）
            progress_callback: 进度回调函数，接收 DownloadProgress 对象

        返回:
            Path — 下载后的模型本地路径

        异常:
            ValueError  — 不支持的模型名
            RuntimeError — 下载失败
        """
        if model_name not in WHISPER_MODELS:
            raise ValueError(
                f"不支持的模型: {model_name}，"
                f"可选: {list(WHISPER_MODELS.keys())}"
            )

        meta = WHISPER_MODELS[model_name]
        repo_id = meta["repo"]

        logger.info(f"[MODEL] 开始下载模型: {model_name} (from {repo_id})")

        self._cancel_event.clear()

        # 设置 HF 镜像
        if self.hf_mirror:
            os.environ["HF_ENDPOINT"] = self.hf_mirror
            logger.info(f"[MODEL] 使用 HF 镜像: {self.hf_mirror}")

        try:
            from huggingface_hub import snapshot_download

            # 准备进度回调
            progress = DownloadProgress(
                model_name=model_name,
                total_bytes=meta["size_mb"] * 1024 * 1024,  # 预估
                status="downloading",
            )

            if progress_callback:
                progress_callback(progress)

            # snapshot_download 会下载整个仓库到本地缓存，
            # 然后我们指定 local_dir 让它直接放到 models/ 下
            local_dir = self.models_dir / f"faster-whisper-{model_name}"

            download_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
            )

            # 验证下载结果
            download_path = Path(download_path)
            if not self._validate_model(download_path):
                raise RuntimeError(
                    f"模型下载完成但验证失败: 缺少必要文件 {MODEL_REQUIRED_FILES}"
                )

            # 完成回调
            progress.status = "completed"
            progress.downloaded_bytes = progress.total_bytes
            if progress_callback:
                progress_callback(progress)

            actual_size = self._get_dir_size_mb(download_path)
            logger.info(
                f"[MODEL] 模型下载完成: {model_name} "
                f"({actual_size:.0f} MB) -> {download_path}"
            )
            return download_path

        except Exception as e:
            if self._cancel_event.is_set():
                logger.info(f"[MODEL] 下载已取消: {model_name}")
                progress = DownloadProgress(
                    model_name=model_name, status="cancelled"
                )
                if progress_callback:
                    progress_callback(progress)
                raise RuntimeError("下载已取消") from e
            else:
                logger.error(f"[MODEL] 下载失败: {model_name} -> {e}")
                progress = DownloadProgress(
                    model_name=model_name, status="error"
                )
                if progress_callback:
                    progress_callback(progress)
                raise RuntimeError(f"模型下载失败 [{model_name}]: {e}") from e
        finally:
            # 清理镜像环境变量
            if self.hf_mirror and "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]

    def cancel_download(self) -> None:
        """取消正在进行的下载"""
        self._cancel_event.set()
        logger.info("[MODEL] 已发送下载取消信号")

    # ── 模型删除 ──────────────────────────────────────────────────────

    def delete_model(self, model_name: str) -> bool:
        """
        删除本地已下载的模型。

        返回:
            bool — 是否成功删除
        """
        local_path = self._find_model_path(model_name)
        if not local_path:
            logger.warning(f"[MODEL] 要删除的模型未找到: {model_name}")
            return False

        try:
            size_mb = self._get_dir_size_mb(local_path)
            shutil.rmtree(local_path)
            logger.info(
                f"[MODEL] 已删除模型: {model_name} "
                f"({size_mb:.0f} MB) -> {local_path}"
            )
            return True
        except Exception as e:
            logger.error(f"[MODEL] 删除模型失败: {model_name} -> {e}")
            return False

    # ── 模型路径解析 ──────────────────────────────────────────────────

    def get_model_path_for_engine(self, model_name: str) -> str:
        """
        获取模型路径，供 faster-whisper 引擎使用。

        如果本地有模型，返回本地路径；
        如果本地没有，返回模型名（让 faster-whisper 自动从 HF 下载）。
        """
        local_path = self._find_model_path(model_name)
        if local_path:
            return str(local_path)
        return model_name

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _find_model_path(self, model_name: str) -> Optional[Path]:
        """
        在 models/ 下查找模型目录。
        支持多种命名格式的识别。
        """
        # 可能的目录名
        candidates = [
            f"faster-whisper-{model_name}",  # HF 下载格式
            f"whisper-{model_name}",
            model_name,                       # 用户可能直接用模型名
        ]

        for dirname in candidates:
            path = self.models_dir / dirname
            if path.is_dir() and self._validate_model(path):
                return path

        # 也检查 huggingface_hub 的缓存目录格式
        # models--Systran--faster-whisper-base/snapshots/xxxxx/
        hf_cache_pattern = f"models--Systran--faster-whisper-{model_name}"
        hf_cache_dir = self.models_dir / hf_cache_pattern
        if hf_cache_dir.is_dir():
            snapshots_dir = hf_cache_dir / "snapshots"
            if snapshots_dir.is_dir():
                # 取最新的 snapshot
                snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for snap in snapshots:
                    if snap.is_dir() and self._validate_model(snap):
                        return snap

        return None

    def _validate_model(self, model_dir: Path) -> bool:
        """验证目录是否包含有效的 CTranslate2 模型文件"""
        if not model_dir.is_dir():
            return False
        for required_file in MODEL_REQUIRED_FILES:
            if not (model_dir / required_file).is_file():
                return False
        return True

    def _get_dir_size_mb(self, path: Path) -> float:
        """计算目录总大小 (MB)"""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except Exception:
            pass
        return total / 1024 / 1024
