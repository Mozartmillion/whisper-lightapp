"""
config_manager.py — LWF 配置管理

负责 config.json 的读写和默认值管理。
首次运行时自动生成默认配置文件。
"""

import json
import logging
from pathlib import Path
from typing import Any

from constants import DEFAULT_CONFIG, APP_VERSION

logger = logging.getLogger("LWF")


class ConfigManager:
    """
    JSON 配置文件读写器。

    使用方式：
        cfg = ConfigManager(config_path)
        value = cfg.get("default_model")
        cfg.set("default_model", "small")
        cfg.save()
    """

    def __init__(self, config_path: Path | str):
        self.path = Path(config_path)
        self._data: dict[str, Any] = {}
        self._load()

    # ── 公开 API ──────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，不存在则返回 default（优先用 DEFAULT_CONFIG 里的值）"""
        if key in self._data:
            return self._data[key]
        if key in DEFAULT_CONFIG:
            return DEFAULT_CONFIG[key]
        return default

    def set(self, key: str, value: Any) -> None:
        """设置配置值（仅内存，需调用 save() 持久化）"""
        self._data[key] = value

    def save(self) -> None:
        """将当前配置写入文件"""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            logger.debug(f"[CONFIG] 配置已保存: {self.path}")
        except Exception as e:
            logger.error(f"[CONFIG] 保存配置失败: {e}")

    def reset(self) -> None:
        """重置为默认配置"""
        self._data = DEFAULT_CONFIG.copy()
        self.save()

    def get_all(self) -> dict[str, Any]:
        """返回完整配置（合并默认值）"""
        merged = DEFAULT_CONFIG.copy()
        merged.update(self._data)
        return merged

    # ── 便捷属性 ──────────────────────────────────────────────────────

    @property
    def hf_mirror(self) -> str:
        return self.get("hf_mirror", "")

    @property
    def default_model(self) -> str:
        return self.get("default_model", "base")

    @property
    def prefer_gpu(self) -> bool:
        return self.get("prefer_gpu", True)

    @property
    def beam_size(self) -> int:
        return self.get("beam_size", 5)

    @property
    def vad_filter(self) -> bool:
        return self.get("vad_filter", True)

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """加载配置文件，不存在则创建默认配置"""
        if self.path.is_file():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info(f"[CONFIG] 已加载配置: {self.path}")
                # 版本迁移：补充新版本新增的 key
                self._migrate()
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[CONFIG] 配置文件损坏，使用默认值: {e}")
                self._data = DEFAULT_CONFIG.copy()
                self.save()
        else:
            logger.info("[CONFIG] 首次运行，创建默认配置")
            self._data = DEFAULT_CONFIG.copy()
            self.save()

    def _migrate(self) -> None:
        """
        版本迁移：如果用户的 config.json 是旧版本生成的，
        补充新版本新增的配置项（不覆盖用户已有的值）。
        """
        changed = False
        for key, default_val in DEFAULT_CONFIG.items():
            if key not in self._data:
                self._data[key] = default_val
                changed = True
                logger.info(f"[CONFIG] 迁移新增配置项: {key} = {default_val}")

        # 更新版本号
        if self._data.get("version") != APP_VERSION:
            self._data["version"] = APP_VERSION
            changed = True

        if changed:
            self.save()
