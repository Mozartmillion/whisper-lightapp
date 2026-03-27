"""
ui_tab_translate.py - 翻译 Tab 页

两种入口：
1. 转录完成后直接翻译（从转录 Tab 传入 segments）
2. 选择本地 SRT 文件翻译

Phase 3: 左右双栏逐句对照编辑器，译文可手动编辑。
API 配置直接在页面内填写，支持预设服务商快速切换。
"""

import threading
import logging
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core_translate import (
    TranslationEngine,
    TranslateConfig,
    TranslateResult,
    API_PROVIDERS,
    parse_srt,
    build_srt,
    build_srt_from_segments,
)

logger = logging.getLogger("LWF")

# 编辑器行配色
_ROW_COLORS = ("#2b2b2b", "#242424")  # 交替行背景
_HIGHLIGHT_COLOR = "#1a3a5c"           # 选中行高亮


class TranslateTab(ctk.CTkFrame):
    """翻译功能 Tab 页"""

    def __init__(self, master, config, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.config = config

        # 状态
        self.is_running: bool = False
        self.should_cancel: bool = False
        self.worker_thread: threading.Thread | None = None

        # 数据源（两种来源）
        self.segments_from_transcribe = None   # list[SegmentInfo]，从转录 Tab 传入
        self.srt_segments: list[dict] | None = None        # 从 SRT 文件解析
        self.srt_file_path: str = ""
        self.translate_results: list[TranslateResult] = []
        self.source_texts: list[str] = []      # 当前要翻译的原文列表

        # 编辑器行组件
        self._editor_rows: list[dict] = []
        # 每行: {"source_label": CTkLabel, "target_entry": CTkEntry,
        #         "left_row_frame": CTkFrame, "right_row_frame": CTkFrame}
        self._selected_row: int = -1

        self._build_ui()
        self._load_config()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI 构建
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)  # 编辑器区域占满
        self.grid_columnconfigure(0, weight=1)

        # ── 区域1: API 配置 ──
        self._build_api_section()

        # ── 区域2: 翻译控制 ──
        self._build_control_section()

        # ── 区域3: 双栏编辑器 ──
        self._build_editor_section()

        # ── 区域4: 底部操作栏 ──
        self._build_bottom_bar()

    def _build_api_section(self):
        """API 配置区域"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        frame.grid_columnconfigure(1, weight=1)

        # 行1: 服务商选择 + 端点
        ctk.CTkLabel(frame, text="服务商:", width=60, anchor="e").grid(
            row=0, column=0, padx=(8, 4), pady=4, sticky="e"
        )

        provider_names = list(API_PROVIDERS.keys())
        self.provider_var = ctk.StringVar(value=provider_names[0])
        self.provider_menu = ctk.CTkOptionMenu(
            frame, variable=self.provider_var,
            values=provider_names,
            command=self._on_provider_changed,
            width=140,
        )
        self.provider_menu.grid(row=0, column=1, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(frame, text="端点:", width=40, anchor="e").grid(
            row=0, column=2, padx=(12, 4), pady=4, sticky="e"
        )
        self.api_base_entry = ctk.CTkEntry(frame, placeholder_text="https://api.deepseek.com/v1")
        self.api_base_entry.grid(row=0, column=3, padx=(4, 8), pady=4, sticky="ew")
        frame.grid_columnconfigure(3, weight=1)

        # 行2: API Key + 模型
        ctk.CTkLabel(frame, text="API Key:", width=60, anchor="e").grid(
            row=1, column=0, padx=(8, 4), pady=4, sticky="e"
        )
        self.api_key_entry = ctk.CTkEntry(frame, placeholder_text="sk-...", show="*")
        self.api_key_entry.grid(row=1, column=1, columnspan=1, padx=4, pady=4, sticky="ew")

        ctk.CTkLabel(frame, text="模型:", width=40, anchor="e").grid(
            row=1, column=2, padx=(12, 4), pady=4, sticky="e"
        )
        self.model_var = ctk.StringVar(value="deepseek-chat")
        self.model_menu = ctk.CTkOptionMenu(
            frame, variable=self.model_var,
            values=["deepseek-chat", "deepseek-reasoner"],
            width=160,
        )
        self.model_menu.grid(row=1, column=3, padx=(4, 8), pady=4, sticky="ew")

        # 测试连接按钮
        self.test_btn = ctk.CTkButton(
            frame, text="测试连接", width=90,
            command=self._on_test_connection,
            fg_color="gray30", hover_color="gray40",
        )
        self.test_btn.grid(row=0, column=4, rowspan=2, padx=(4, 8), pady=4)

    def _build_control_section(self):
        """翻译控制区域"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        frame.grid_columnconfigure(2, weight=1)

        # 左侧: 文件源
        source_frame = ctk.CTkFrame(frame, fg_color="transparent")
        source_frame.grid(row=0, column=0, padx=8, pady=4, sticky="w")

        self.source_label = ctk.CTkLabel(
            source_frame, text="无数据源",
            text_color="gray", font=ctk.CTkFont(size=12),
        )
        self.source_label.pack(side="left", padx=(0, 8))

        self.load_srt_btn = ctk.CTkButton(
            source_frame, text="选择 SRT 文件", width=120,
            command=self._on_load_srt,
            fg_color="gray30", hover_color="gray40",
        )
        self.load_srt_btn.pack(side="left", padx=4)

        # 中间: 目标语言
        lang_frame = ctk.CTkFrame(frame, fg_color="transparent")
        lang_frame.grid(row=0, column=1, padx=8, pady=4)

        ctk.CTkLabel(lang_frame, text="目标语言:").pack(side="left", padx=(0, 4))
        self.target_lang_var = ctk.StringVar(value="简体中文")
        target_langs = [
            "简体中文", "繁體中文", "English", "日本語",
            "한국어", "Français", "Deutsch", "Español",
        ]
        self.target_lang_menu = ctk.CTkOptionMenu(
            lang_frame, variable=self.target_lang_var,
            values=target_langs, width=120,
        )
        self.target_lang_menu.pack(side="left")

        # 右侧: 双语 + 翻译按钮
        right_frame = ctk.CTkFrame(frame, fg_color="transparent")
        right_frame.grid(row=0, column=2, padx=8, pady=4, sticky="e")

        self.bilingual_var = ctk.BooleanVar(value=False)
        self.bilingual_check = ctk.CTkCheckBox(
            right_frame, text="双语字幕",
            variable=self.bilingual_var,
        )
        self.bilingual_check.pack(side="left", padx=(0, 12))

        self.translate_btn = ctk.CTkButton(
            right_frame, text="▶ 开始翻译", width=120,
            command=self._on_translate,
            fg_color="#28a745", hover_color="#218838",
        )
        self.translate_btn.pack(side="left", padx=4)

        self.cancel_btn = ctk.CTkButton(
            right_frame, text="取消", width=70,
            command=self._on_cancel,
            fg_color="#dc3545", hover_color="#c82333",
            state="disabled",
        )
        self.cancel_btn.pack(side="left", padx=4)

    def _build_editor_section(self):
        """双栏逐句对照编辑器"""
        editor_frame = ctk.CTkFrame(self)
        editor_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        editor_frame.grid_rowconfigure(1, weight=1)
        editor_frame.grid_columnconfigure(0, weight=1)

        # ── 进度条 + 状态 ──
        progress_frame = ctk.CTkFrame(editor_frame, fg_color="transparent")
        progress_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        progress_frame.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(progress_frame)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.progress.set(0)

        self.status_var = ctk.StringVar(value="就绪")
        self.status_label = ctk.CTkLabel(
            progress_frame, textvariable=self.status_var,
            font=ctk.CTkFont(size=11), text_color="gray",
            width=200, anchor="e",
        )
        self.status_label.grid(row=0, column=1, sticky="e")

        # ── 列标题 ──
        header_frame = ctk.CTkFrame(editor_frame, fg_color="transparent", height=28)
        header_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header_frame, text="原文",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#888", anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(4, 0))

        ctk.CTkLabel(
            header_frame, text="译文（可编辑）",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#888", anchor="w",
        ).grid(row=0, column=1, sticky="w", padx=(4, 0))

        # ── 双栏容器（用一个 scrollable frame 包两列）──
        self.editor_scroll = ctk.CTkScrollableFrame(
            editor_frame, fg_color="transparent",
        )
        self.editor_scroll.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 8))
        editor_frame.grid_rowconfigure(2, weight=1)

        # 两列等宽
        self.editor_scroll.grid_columnconfigure(0, weight=1)
        self.editor_scroll.grid_columnconfigure(1, weight=1)

        # 初始提示
        self._hint_label = ctk.CTkLabel(
            self.editor_scroll,
            text="翻译结果将显示在这里...\n\n"
                 "支持两种方式：\n"
                 "1. 转录完成后点击「翻译字幕」自动导入\n"
                 "2. 点击「选择 SRT 文件」加载本地字幕",
            text_color="gray", font=ctk.CTkFont(size=13),
            justify="left",
        )
        self._hint_label.grid(row=0, column=0, columnspan=2, pady=20, padx=20, sticky="w")

    def _build_bottom_bar(self):
        """底部操作栏"""
        bar = ctk.CTkFrame(self)
        bar.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

        self.save_srt_btn = ctk.CTkButton(
            bar, text="保存翻译 SRT", width=130,
            command=self._on_save_translated_srt,
            state="disabled",
        )
        self.save_srt_btn.pack(side="left", padx=8, pady=4)

        self.save_bilingual_btn = ctk.CTkButton(
            bar, text="保存双语 SRT", width=130,
            command=self._on_save_bilingual_srt,
            state="disabled",
        )
        self.save_bilingual_btn.pack(side="left", padx=4, pady=4)

        self.copy_btn = ctk.CTkButton(
            bar, text="复制译文", width=100,
            command=self._on_copy,
            state="disabled",
            fg_color="gray30", hover_color="gray40",
        )
        self.copy_btn.pack(side="left", padx=4, pady=4)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 编辑器操作
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _populate_source_only(self):
        """加载数据后立即填充左栏原文，右栏 Entry 留空等翻译"""
        self._clear_editor()

        for i, text in enumerate(self.source_texts):
            row_color = _ROW_COLORS[i % 2]

            # 获取时间轴
            timecode = ""
            if self.segments_from_transcribe and i < len(self.segments_from_transcribe):
                seg = self.segments_from_transcribe[i]
                timecode = f"[{seg.start_fmt} -> {seg.end_fmt}]"
            elif self.srt_segments and i < len(self.srt_segments):
                seg = self.srt_segments[i]
                timecode = f"[{seg['start']} -> {seg['end']}]"

            header_text = f"#{i+1}  {timecode}" if timecode else f"#{i+1}"

            # ── 左栏 ──
            left_frame = ctk.CTkFrame(
                self.editor_scroll, fg_color=row_color, corner_radius=0,
            )
            left_frame.grid(row=i, column=0, sticky="ew", padx=(0, 2), pady=0)
            left_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                left_frame, text=header_text,
                font=ctk.CTkFont(size=10), text_color="#666", anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=6, pady=(4, 0))

            source_label = ctk.CTkLabel(
                left_frame, text=text,
                font=ctk.CTkFont(size=13), anchor="w",
                wraplength=400, justify="left",
            )
            source_label.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))

            # ── 右栏 ──
            right_frame = ctk.CTkFrame(
                self.editor_scroll, fg_color=row_color, corner_radius=0,
            )
            right_frame.grid(row=i, column=1, sticky="ew", padx=(2, 0), pady=0)
            right_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                right_frame, text=header_text,
                font=ctk.CTkFont(size=10), text_color="#666", anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=6, pady=(4, 0))

            target_entry = ctk.CTkEntry(
                right_frame, font=ctk.CTkFont(size=13), height=32,
                placeholder_text="等待翻译...",
            )
            target_entry.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))

            # 行点击高亮
            row_idx = i
            for widget in [left_frame, source_label]:
                widget.bind("<Button-1>",
                    lambda e, idx=row_idx: self._on_row_click(idx))
            target_entry.bind("<FocusIn>",
                lambda e, idx=row_idx: self._on_row_click(idx))

            self._editor_rows.append({
                "source_label": source_label,
                "target_entry": target_entry,
                "left_frame": left_frame,
                "right_frame": right_frame,
                "bg_color": row_color,
            })

        self.status_var.set(f"已加载 {len(self.source_texts)} 句，可开始翻译")

    def _populate_editor(self, results: list[TranslateResult]):
        """翻译完成后填充双栏编辑器"""
        self._clear_editor()

        for i, r in enumerate(results):
            row_color = _ROW_COLORS[i % 2]

            # 获取时间轴
            timecode = ""
            if self.segments_from_transcribe and i < len(self.segments_from_transcribe):
                seg = self.segments_from_transcribe[i]
                timecode = f"[{seg.start_fmt} -> {seg.end_fmt}]"
            elif self.srt_segments and i < len(self.srt_segments):
                seg = self.srt_segments[i]
                timecode = f"[{seg['start']} -> {seg['end']}]"

            # ── 左栏：序号 + 时间轴 + 原文（只读）──
            left_frame = ctk.CTkFrame(
                self.editor_scroll, fg_color=row_color,
                corner_radius=0,
            )
            left_frame.grid(row=i, column=0, sticky="ew", padx=(0, 2), pady=0)
            left_frame.grid_columnconfigure(0, weight=1)

            header_text = f"#{i+1}  {timecode}" if timecode else f"#{i+1}"
            ctk.CTkLabel(
                left_frame, text=header_text,
                font=ctk.CTkFont(size=10),
                text_color="#666", anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=6, pady=(4, 0))

            source_label = ctk.CTkLabel(
                left_frame, text=r.source,
                font=ctk.CTkFont(size=13), anchor="w",
                wraplength=400, justify="left",
            )
            source_label.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))

            # ── 右栏：序号 + 时间轴 + 译文 Entry（可编辑）──
            right_frame = ctk.CTkFrame(
                self.editor_scroll, fg_color=row_color,
                corner_radius=0,
            )
            right_frame.grid(row=i, column=1, sticky="ew", padx=(2, 0), pady=0)
            right_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                right_frame, text=header_text,
                font=ctk.CTkFont(size=10),
                text_color="#666", anchor="w",
            ).grid(row=0, column=0, sticky="w", padx=6, pady=(4, 0))

            target_entry = ctk.CTkEntry(
                right_frame,
                font=ctk.CTkFont(size=13),
                height=32,
            )
            target_entry.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 4))
            target_entry.insert(0, r.target if r.target else "")

            # 行点击高亮
            row_idx = i
            for widget in [left_frame, source_label]:
                widget.bind("<Button-1>",
                    lambda e, idx=row_idx: self._on_row_click(idx))
            target_entry.bind("<FocusIn>",
                lambda e, idx=row_idx: self._on_row_click(idx))

            # 保存引用
            self._editor_rows.append({
                "source_label": source_label,
                "target_entry": target_entry,
                "left_frame": left_frame,
                "right_frame": right_frame,
                "bg_color": row_color,
            })

    def _on_row_click(self, row_idx: int):
        """点击行高亮"""
        # 取消上一行高亮
        if 0 <= self._selected_row < len(self._editor_rows):
            prev = self._editor_rows[self._selected_row]
            prev["left_frame"].configure(fg_color=prev["bg_color"])
            prev["right_frame"].configure(fg_color=prev["bg_color"])

        # 高亮当前行
        if 0 <= row_idx < len(self._editor_rows):
            curr = self._editor_rows[row_idx]
            curr["left_frame"].configure(fg_color=_HIGHLIGHT_COLOR)
            curr["right_frame"].configure(fg_color=_HIGHLIGHT_COLOR)
            self._selected_row = row_idx

    def _fill_batch_translations(self, batch_start: int, batch_end: int,
                                  batch_results: list[TranslateResult]):
        """逐批将译文填入右栏 Entry（由 batch_done_callback 触发，主线程执行）"""
        for i, r in enumerate(batch_results):
            row_idx = batch_start + i
            if 0 <= row_idx < len(self._editor_rows):
                entry = self._editor_rows[row_idx]["target_entry"]
                # 翻译中 Entry 处于 disabled 状态，需临时解锁才能写入
                entry.configure(state="normal")
                entry.delete(0, "end")
                entry.insert(0, r.target if r.target else "")
                # 移除 placeholder 视觉效果
                if r.target:
                    entry.configure(placeholder_text="")
                # 写入后重新禁用，防止翻译期间用户误编辑
                if self.is_running:
                    entry.configure(state="disabled")

    def _clear_editor(self):
        """清空编辑器所有行"""
        for row in self._editor_rows:
            row["left_frame"].destroy()
            row["right_frame"].destroy()
        self._editor_rows.clear()
        self._selected_row = -1

        # 移除提示标签（如果还在）
        if hasattr(self, "_hint_label") and self._hint_label.winfo_exists():
            self._hint_label.destroy()

    def _collect_translations_from_editor(self) -> list[TranslateResult]:
        """从编辑器 Entry 收集用户可能修改过的译文"""
        results = []
        for i, row in enumerate(self._editor_rows):
            target = row["target_entry"].get().strip()
            source = self.source_texts[i] if i < len(self.source_texts) else ""
            results.append(TranslateResult(
                index=i,
                source=source,
                target=target,
                success=bool(target),
            ))
        return results

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 外部接口（供转录 Tab / main_ui 调用）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def load_from_transcribe(self, segments, source_lang: str = ""):
        """
        从转录 Tab 传入 segments，准备翻译。
        """
        self.segments_from_transcribe = segments
        self.srt_segments = None
        self.srt_file_path = ""
        self.source_texts = [seg.text.strip() for seg in segments]

        count = len(segments)
        self.source_label.configure(
            text=f"来源: 转录结果 ({count} 句)",
            text_color="white",
        )
        logger.info(f"[TRANSLATE-UI] 从转录导入 {count} 句")
        self._clear_results()
        # 立即填充左栏预览
        self._populate_source_only()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 配置管理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _load_config(self):
        """从配置文件恢复 API 设置"""
        provider = self.config.get("translate_provider", "DeepSeek")
        api_base = self.config.get("translate_api_base", "")
        api_key = self.config.get("translate_api_key", "")
        model = self.config.get("translate_model", "")
        target_lang = self.config.get("translate_target_lang", "简体中文")

        if provider in API_PROVIDERS:
            self.provider_var.set(provider)
            self._on_provider_changed(provider)

        if api_base:
            self.api_base_entry.delete(0, "end")
            self.api_base_entry.insert(0, api_base)
        if api_key:
            self.api_key_entry.delete(0, "end")
            self.api_key_entry.insert(0, api_key)
        if model:
            self.model_var.set(model)

        self.target_lang_var.set(target_lang)

    def _save_config(self):
        """保存 API 设置到配置文件"""
        self.config.set("translate_provider", self.provider_var.get())
        self.config.set("translate_api_base", self.api_base_entry.get().strip())
        self.config.set("translate_api_key", self.api_key_entry.get().strip())
        self.config.set("translate_model", self.model_var.get())
        self.config.set("translate_target_lang", self.target_lang_var.get())
        self.config.save()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 事件处理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_provider_changed(self, provider_name: str):
        """切换服务商时更新端点和模型列表"""
        if provider_name not in API_PROVIDERS:
            return

        provider = API_PROVIDERS[provider_name]

        self.api_base_entry.delete(0, "end")
        if provider["api_base"]:
            self.api_base_entry.insert(0, provider["api_base"])

        models = provider["models"] if provider["models"] else [""]
        self.model_menu.configure(values=models if models else ["自定义模型"])
        if provider["default_model"]:
            self.model_var.set(provider["default_model"])
        elif models:
            self.model_var.set(models[0])

    def _on_test_connection(self):
        """测试 API 连接"""
        cfg = self._get_translate_config()
        if not cfg.api_base:
            messagebox.showwarning("提示", "请先填写 API 端点")
            return

        self.test_btn.configure(state="disabled", text="测试中...")

        def _test():
            engine = TranslationEngine(cfg)
            success, msg = engine.test_connection()
            self.after(0, lambda: self._on_test_done(success, msg))

        threading.Thread(target=_test, daemon=True).start()

    def _on_test_done(self, success: bool, msg: str):
        self.test_btn.configure(state="normal", text="测试连接")
        if success:
            self.status_var.set("API 连接成功")
            messagebox.showinfo("连接测试", msg)
        else:
            self.status_var.set("API 连接失败")
            messagebox.showerror("连接测试", msg)

    def _on_load_srt(self):
        """加载本地 SRT 文件"""
        file_path = filedialog.askopenfilename(
            title="选择 SRT 字幕文件",
            filetypes=[("SRT 字幕", "*.srt"), ("所有文件", "*.*")],
        )
        if not file_path:
            return

        try:
            srt_text = Path(file_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                srt_text = Path(file_path).read_text(encoding="gbk")
            except Exception as e:
                messagebox.showerror("错误", f"无法读取文件: {e}")
                return

        segments = parse_srt(srt_text)
        if not segments:
            messagebox.showwarning("提示", "未从文件中解析到字幕内容")
            return

        self.srt_segments = segments
        self.segments_from_transcribe = None
        self.srt_file_path = file_path
        self.source_texts = [seg["text"] for seg in segments]

        name = Path(file_path).name
        count = len(segments)
        self.source_label.configure(
            text=f"来源: {name} ({count} 句)",
            text_color="white",
        )
        logger.info(f"[TRANSLATE-UI] 加载 SRT: {name}, {count} 句")
        self._clear_results()
        # 立即填充左栏预览
        self._populate_source_only()

    def _on_translate(self):
        """开始翻译"""
        if not self.source_texts:
            messagebox.showwarning("提示", "请先加载字幕数据（从转录导入或选择 SRT 文件）")
            return

        cfg = self._get_translate_config()
        if not cfg.api_base:
            messagebox.showwarning("提示", "请填写 API 端点")
            return

        self._save_config()
        self._set_running(True)

        # 如果编辑器还没填充（直接点翻译没先加载预览），先填左栏
        if not self._editor_rows:
            self._populate_source_only()

        # 清空右栏（重新翻译时）
        for row in self._editor_rows:
            row["target_entry"].delete(0, "end")
            row["target_entry"].configure(placeholder_text="翻译中...")

        self.translate_results = []
        self.progress.set(0)

        def _worker():
            try:
                engine = TranslationEngine(cfg)
                results = engine.translate_batch(
                    texts=self.source_texts,
                    progress_callback=lambda done, total, preview:
                        self.after(0, lambda d=done, t=total, p=preview:
                            self._ui_on_progress(d, t, p)),
                    batch_done_callback=lambda bs, be, br:
                        self.after(0, lambda s=bs, e=be, r=list(br):
                            self._fill_batch_translations(s, e, r)),
                    cancel_check=lambda: self.should_cancel,
                )
                self.after(0, lambda r=results: self._ui_on_done(r))

            except Exception as e:
                logger.exception("[TRANSLATE] 翻译出错")
                self.after(0, lambda err=str(e): self._ui_on_error(err))

        self.worker_thread = threading.Thread(target=_worker, daemon=True)
        self.worker_thread.start()

    def _on_cancel(self):
        self.should_cancel = True
        self.status_var.set("正在取消...")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI 更新回调（主线程）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _ui_on_progress(self, completed: int, total: int, preview: str):
        ratio = completed / total if total > 0 else 0
        self.progress.set(ratio)
        self.status_var.set(f"翻译中 {completed}/{total}  {preview}")

    def _ui_on_done(self, results: list[TranslateResult]):
        """翻译完成 → 收尾（编辑器已逐批填充完毕）"""
        self.translate_results = results
        success_count = sum(1 for r in results if r.success)
        total = len(results)

        self.progress.set(1.0)
        if self.should_cancel:
            self.status_var.set(f"已取消 (已翻译 {success_count}/{total} 句)")
        else:
            self.status_var.set(
                f"翻译完成！{success_count}/{total} 句成功，可直接编辑译文"
            )

        # 启用保存按钮
        self.save_srt_btn.configure(state="normal")
        self.save_bilingual_btn.configure(state="normal")
        self.copy_btn.configure(state="normal")

        self._set_running(False)

    def _ui_on_error(self, error_msg: str):
        self.status_var.set(f"翻译出错: {error_msg[:80]}")
        self.progress.set(0)
        self._set_running(False)
        messagebox.showerror("翻译错误", error_msg)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 导出（从编辑器 Entry 收集最新译文）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_save_translated_srt(self):
        self._save_srt(bilingual=False)

    def _on_save_bilingual_srt(self):
        self._save_srt(bilingual=True)

    def _save_srt(self, bilingual: bool):
        if not self._editor_rows:
            messagebox.showinfo("提示", "没有翻译结果可保存")
            return

        # 从编辑器收集最新译文（用户可能手动改过）
        current_results = self._collect_translations_from_editor()

        suffix = "_bilingual.srt" if bilingual else "_translated.srt"

        if self.srt_file_path:
            stem = Path(self.srt_file_path).stem
        else:
            stem = "output"

        default_name = stem + suffix
        save_path = filedialog.asksaveasfilename(
            title="保存字幕文件",
            initialfile=default_name,
            defaultextension=".srt",
            filetypes=[("SRT 字幕", "*.srt")],
        )
        if not save_path:
            return

        if self.srt_segments is not None:
            srt_text = build_srt(
                self.srt_segments, current_results, bilingual=bilingual
            )
        elif self.segments_from_transcribe is not None:
            srt_text = build_srt_from_segments(
                self.segments_from_transcribe, current_results,
                bilingual=bilingual,
            )
        else:
            messagebox.showerror("错误", "数据源丢失")
            return

        Path(save_path).write_text(srt_text, encoding="utf-8")
        self.status_var.set(f"已保存: {Path(save_path).name}")

    def _on_copy(self):
        if not self._editor_rows:
            return
        texts = []
        for row in self._editor_rows:
            t = row["target_entry"].get().strip()
            if t:
                texts.append(t)
        self.clipboard_clear()
        self.clipboard_append("\n".join(texts))
        self.status_var.set("已复制到剪贴板")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 工具
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_translate_config(self) -> TranslateConfig:
        return TranslateConfig(
            api_base=self.api_base_entry.get().strip(),
            api_key=self.api_key_entry.get().strip(),
            model=self.model_var.get(),
            target_lang=self.target_lang_var.get(),
            bilingual=self.bilingual_var.get(),
        )

    def _set_running(self, running: bool):
        self.is_running = running
        self.should_cancel = False

        state_on = "normal"
        state_off = "disabled"

        if running:
            self.translate_btn.configure(state=state_off)
            self.cancel_btn.configure(state=state_on)
            self.load_srt_btn.configure(state=state_off)
            self.provider_menu.configure(state=state_off)
            self.api_base_entry.configure(state=state_off)
            self.api_key_entry.configure(state=state_off)
            self.model_menu.configure(state=state_off)
            self.target_lang_menu.configure(state=state_off)
            self.bilingual_check.configure(state=state_off)
            self.save_srt_btn.configure(state=state_off)
            self.save_bilingual_btn.configure(state=state_off)
            self.copy_btn.configure(state=state_off)
            # 翻译中禁用编辑器
            for row in self._editor_rows:
                row["target_entry"].configure(state=state_off)
        else:
            self.translate_btn.configure(state=state_on)
            self.cancel_btn.configure(state=state_off)
            self.load_srt_btn.configure(state=state_on)
            self.provider_menu.configure(state=state_on)
            self.api_base_entry.configure(state=state_on)
            self.api_key_entry.configure(state=state_on)
            self.model_menu.configure(state=state_on)
            self.target_lang_menu.configure(state=state_on)
            self.bilingual_check.configure(state=state_on)
            # 翻译完启用编辑器
            for row in self._editor_rows:
                row["target_entry"].configure(state=state_on)

    def _clear_results(self):
        self._clear_editor()
        self.translate_results = []
        self.progress.set(0)
        self.status_var.set("就绪")
        self.save_srt_btn.configure(state="disabled")
        self.save_bilingual_btn.configure(state="disabled")
        self.copy_btn.configure(state="disabled")
