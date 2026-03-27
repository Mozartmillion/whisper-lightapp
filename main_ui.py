"""
main_ui.py — LWF v2.0 GUI 入口

基于 CustomTkinter 构建的 Tab 页主界面。
启动时首先调用 utils_env 完成环境初始化。

Tab 页结构：
  - 🎙 转录：文件选择 → 参数设置 → 转录 → 结果导出
  - 🌐 翻译：API 配置 → 字幕翻译 → 双语/译文导出
  - 📦 模型管理：模型列表 → 下载 / 删除
  - ⚙ 设置：镜像地址、默认参数、关于信息
"""

# ⚠ 环境初始化必须在所有其他 import 之前执行
from utils_env import init_environment, clean_temp
env = init_environment()

import logging
import customtkinter as ctk

from constants import APP_TITLE
from config_manager import ConfigManager
from model_manager import ModelManager
from ui_tab_transcribe import TranscribeTab
from ui_tab_models import ModelsTab
from ui_tab_settings import SettingsTab
from ui_tab_translate import TranslateTab

logger = logging.getLogger("LWF")

# ═══════════════════════════════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════════════════════════════
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

WINDOW_SIZE = "1000x720"
MIN_WIDTH = 800
MIN_HEIGHT = 580


# ═══════════════════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════════════════

class LWFApp(ctk.CTk):
    """LWF v2.0 主窗口"""

    def __init__(self):
        super().__init__()

        # ── 窗口基础设置 ──
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(MIN_WIDTH, MIN_HEIGHT)

        # ── 初始化配置和模型管理器 ──
        self.config = ConfigManager(env.config_path)
        self.model_manager = ModelManager(
            models_dir=env.models_dir,
            hf_mirror=self.config.hf_mirror,
        )

        # ── 构建界面 ──
        self._build_ui()

        # ── 启动检查 ──
        self._startup_check()

    def _build_ui(self):
        """构建 Tab 页主界面"""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Tab 页容器 ──
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))

        # 创建四个 Tab
        tab_transcribe = self.tabview.add("🎙 转录")
        tab_translate = self.tabview.add("🌐 翻译")
        tab_models = self.tabview.add("📦 模型管理")
        tab_settings = self.tabview.add("⚙ 设置")

        # 让每个 Tab 内容区填满
        for tab in [tab_transcribe, tab_translate, tab_models, tab_settings]:
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)

        # ── 实例化各 Tab ──
        self.transcribe_tab = TranscribeTab(
            tab_transcribe, env=env, config=self.config,
            model_manager=self.model_manager,
        )
        self.transcribe_tab.grid(row=0, column=0, sticky="nsew")

        self.translate_tab = TranslateTab(
            tab_translate, config=self.config,
        )
        self.translate_tab.grid(row=0, column=0, sticky="nsew")

        # 把翻译 Tab 引用传给转录 Tab，用于联动
        self.transcribe_tab.translate_tab = self.translate_tab
        self.transcribe_tab.tabview = self.tabview

        self.models_tab = ModelsTab(
            tab_models, model_manager=self.model_manager,
            on_model_changed=self._on_model_changed,
        )
        self.models_tab.grid(row=0, column=0, sticky="nsew")

        self.settings_tab = SettingsTab(
            tab_settings, config=self.config, env=env,
        )
        self.settings_tab.grid(row=0, column=0, sticky="nsew")

        # ── 底部状态栏 ──
        self._build_status_bar()

    def _build_status_bar(self):
        """底部环境状态栏"""
        bar = ctk.CTkFrame(self, height=30)
        bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        gpu_icon = "🟢 GPU" if env.gpu_available else "⚪ CPU"
        ffmpeg_icon = "🟢 ffmpeg" if env.ffmpeg_path else "🔴 ffmpeg"

        installed = self.model_manager.get_installed_model_names()
        model_text = f"模型: {len(installed)} 个" if installed else "模型: 未安装"

        self.status_label = ctk.CTkLabel(
            bar,
            text=f"  {gpu_icon}  │  {ffmpeg_icon}  │  {model_text}",
            font=ctk.CTkFont(size=11), text_color="gray", anchor="w",
        )
        self.status_label.pack(side="left", padx=8, pady=4)

    def _on_model_changed(self):
        """模型列表变更时更新相关 UI"""
        # 刷新转录 Tab 的模型下拉框
        self.transcribe_tab.refresh_model_list()

        # 刷新状态栏
        installed = self.model_manager.get_installed_model_names()
        model_text = f"模型: {len(installed)} 个" if installed else "模型: 未安装"
        gpu_icon = "🟢 GPU" if env.gpu_available else "⚪ CPU"
        ffmpeg_icon = "🟢 ffmpeg" if env.ffmpeg_path else "🔴 ffmpeg"
        self.status_label.configure(
            text=f"  {gpu_icon}  │  {ffmpeg_icon}  │  {model_text}"
        )

        # 同步镜像配置到 model_manager
        self.model_manager.hf_mirror = self.config.hf_mirror

    def _startup_check(self):
        """启动时检查"""
        # 检查 ffmpeg
        if not env.ffmpeg_path:
            self.after(300, lambda: self._show_startup_warning(
                "缺少 FFmpeg",
                "未检测到 ffmpeg！\n\n"
                "请将 ffmpeg.exe 和 ffprobe.exe 放入程序目录下的 /bin 文件夹中。\n"
                "否则将无法处理视频/音频文件。"
            ))

        # 检查模型
        installed = self.model_manager.get_installed_model_names()
        if not installed:
            self.after(600, lambda: self._guide_to_models_tab())

    def _show_startup_warning(self, title: str, message: str):
        """显示启动警告"""
        from tkinter import messagebox
        messagebox.showwarning(title, message)

    def _guide_to_models_tab(self):
        """引导用户到模型管理 Tab"""
        from tkinter import messagebox
        result = messagebox.showinfo(
            "欢迎使用 LWF",
            "检测到尚未安装任何 Whisper 模型。\n\n"
            "即将跳转到「模型管理」页面，请先下载一个模型。\n"
            "推荐初次使用选择 base 模型（~145 MB）。"
        )
        self.tabview.set("📦 模型管理")


# ═══════════════════════════════════════════════════════════════════════════
# 程序入口
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = LWFApp()
    app.mainloop()
