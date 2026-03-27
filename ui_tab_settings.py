"""
ui_tab_settings.py — 设置 Tab 页

功能：
  - HuggingFace 镜像地址配置
  - 默认参数设置
  - 关于信息
"""

import logging
from tkinter import messagebox

import customtkinter as ctk

from constants import APP_NAME, APP_VERSION, WHISPER_MODELS, DEFAULT_CONFIG
from config_manager import ConfigManager

logger = logging.getLogger("LWF")


class SettingsTab(ctk.CTkFrame):
    """设置 Tab 页"""

    def __init__(self, master, config: ConfigManager, env, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config
        self.env = env
        self._build_ui()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI 构建
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_ui(self):
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_mirror_section()
        self._build_default_params_section()
        self._build_about_section()
        self._build_action_section()

    def _build_mirror_section(self):
        """HuggingFace 镜像设置"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="🌐 模型下载源",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 4), sticky="w")

        ctk.CTkLabel(
            frame, text="HuggingFace 镜像:",
            font=ctk.CTkFont(size=13),
        ).grid(row=1, column=0, padx=12, pady=6, sticky="w")

        self.mirror_var = ctk.StringVar(value=self.config.hf_mirror)
        self.mirror_entry = ctk.CTkEntry(
            frame, textvariable=self.mirror_var,
            placeholder_text="留空使用官方源，国内推荐填 https://hf-mirror.com",
            font=ctk.CTkFont(size=12),
        )
        self.mirror_entry.grid(row=1, column=1, padx=(4, 12), pady=6, sticky="ew")

        ctk.CTkLabel(
            frame,
            text="💡 国内用户如果下载模型很慢，可以填入 https://hf-mirror.com",
            font=ctk.CTkFont(size=11), text_color="gray",
        ).grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 8), sticky="w")

    def _build_default_params_section(self):
        """默认参数设置"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="ew", padx=12, pady=4)

        ctk.CTkLabel(
            frame, text="⚙ 默认参数",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=4, padx=12, pady=(10, 4), sticky="w")

        # Beam Size
        ctk.CTkLabel(frame, text="Beam Size:", font=ctk.CTkFont(size=13)).grid(
            row=1, column=0, padx=12, pady=6, sticky="w"
        )
        self.beam_var = ctk.StringVar(value=str(self.config.beam_size))
        ctk.CTkOptionMenu(
            frame, variable=self.beam_var,
            values=["1", "3", "5", "8", "10"], width=80,
        ).grid(row=1, column=1, padx=4, pady=6, sticky="w")

        # VAD 开关
        ctk.CTkLabel(frame, text="VAD 静音检测:", font=ctk.CTkFont(size=13)).grid(
            row=1, column=2, padx=(20, 4), pady=6, sticky="w"
        )
        self.vad_var = ctk.BooleanVar(value=self.config.vad_filter)
        ctk.CTkSwitch(
            frame, text="", variable=self.vad_var,
        ).grid(row=1, column=3, padx=4, pady=6, sticky="w")

        # GPU 偏好
        ctk.CTkLabel(frame, text="优先使用 GPU:", font=ctk.CTkFont(size=13)).grid(
            row=2, column=0, padx=12, pady=(6, 10), sticky="w"
        )
        self.prefer_gpu_var = ctk.BooleanVar(value=self.config.prefer_gpu)
        ctk.CTkSwitch(
            frame, text="", variable=self.prefer_gpu_var,
        ).grid(row=2, column=1, padx=4, pady=(6, 10), sticky="w")

    def _build_about_section(self):
        """关于信息"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, sticky="ew", padx=12, pady=4)

        ctk.CTkLabel(
            frame, text="ℹ 关于",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        info_lines = [
            f"应用: {APP_NAME}",
            f"版本: v{APP_VERSION}",
            f"引擎: faster-whisper (CTranslate2)",
            f"UI: CustomTkinter",
            "",
            f"根目录: {self.env.root_dir}",
            f"FFmpeg: {self.env.ffmpeg_path or '未检测到'}",
            f"GPU: {'可用 ✅' if self.env.gpu_available else 'CPU 模式'}",
        ]
        if self.env.gpu_dlls_found:
            info_lines.append(f"GPU DLL: {', '.join(self.env.gpu_dlls_found)}")

        info_text = "\n".join(info_lines)
        info_label = ctk.CTkLabel(
            frame, text=info_text,
            font=ctk.CTkFont(size=12, family="Consolas"),
            justify="left", anchor="w",
        )
        info_label.grid(row=1, column=0, padx=12, pady=(4, 10), sticky="w")

    def _build_action_section(self):
        """底部操作按钮"""
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=4, column=0, sticky="ew", padx=12, pady=(4, 12))

        ctk.CTkButton(
            frame, text="💾 保存设置",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40, command=self._on_save,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            frame, text="↩ 恢复默认",
            height=40, fg_color="gray", hover_color="#555",
            command=self._on_reset,
        ).pack(side="left", padx=4)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 事件处理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_save(self):
        """保存设置"""
        self.config.set("hf_mirror", self.mirror_var.get().strip())
        self.config.set("beam_size", int(self.beam_var.get()))
        self.config.set("vad_filter", self.vad_var.get())
        self.config.set("prefer_gpu", self.prefer_gpu_var.get())
        self.config.save()
        messagebox.showinfo("设置", "设置已保存 ✅")

    def _on_reset(self):
        """恢复默认设置"""
        confirm = messagebox.askyesno("确认", "确认将所有设置恢复为默认值？")
        if not confirm:
            return

        self.config.reset()

        # 刷新 UI
        self.mirror_var.set("")
        self.beam_var.set(str(DEFAULT_CONFIG["beam_size"]))
        self.vad_var.set(DEFAULT_CONFIG["vad_filter"])
        self.prefer_gpu_var.set(DEFAULT_CONFIG["prefer_gpu"])

        messagebox.showinfo("设置", "已恢复默认设置 ✅")
