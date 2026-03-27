"""
ui_tab_transcribe.py — 转录 Tab 页

包含文件选择、参数设置、转录控制、结果展示和导出功能。
从 v1 的 main_ui.py 重构而来，核心逻辑不变，适配 Tab 页容器。
"""

import threading
import time
import logging
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core_transcribe import (
    extract_audio,
    transcribe_stream,
    segments_to_srt,
    segments_to_text,
    SegmentInfo,
)
from constants import LANGUAGE_OPTIONS, FILEDIALOG_TYPES

logger = logging.getLogger("LWF")


class TranscribeTab(ctk.CTkFrame):
    """转录功能 Tab 页"""

    def __init__(self, master, env, config, model_manager, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.env = env
        self.config = config
        self.model_manager = model_manager

        # 状态变量
        self.selected_file: str = ""
        self.is_running: bool = False
        self.should_cancel: bool = False
        self.worker_thread: threading.Thread | None = None
        self.all_segments: list[SegmentInfo] = []
        self.audio_duration: float = 0.0

        self._build_ui()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI 构建
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_ui(self):
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_file_section()
        self._build_param_section()
        self._build_control_section()
        self._build_result_section()
        self._build_bottom_section()

    def _build_file_section(self):
        """文件选择区"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="📂 文件", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, padx=12, pady=10, sticky="w")

        self.file_path_var = ctk.StringVar(value="点击「浏览」选择音视频文件...")
        self.file_entry = ctk.CTkEntry(
            frame, textvariable=self.file_path_var,
            state="readonly", font=ctk.CTkFont(size=13)
        )
        self.file_entry.grid(row=0, column=1, padx=6, pady=10, sticky="ew")

        self.browse_btn = ctk.CTkButton(
            frame, text="浏览", width=80, command=self._on_browse
        )
        self.browse_btn.grid(row=0, column=2, padx=12, pady=10)

    def _build_param_section(self):
        """参数设置区"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, sticky="ew", padx=12, pady=4)

        # 模型选择
        ctk.CTkLabel(frame, text="模型:").grid(
            row=0, column=0, padx=(12, 4), pady=10, sticky="w"
        )

        installed = self.model_manager.get_installed_model_names()
        if not installed:
            installed = ["(无可用模型)"]
        default_model = self.config.default_model
        if default_model not in installed:
            default_model = installed[0] if installed else "(无可用模型)"

        self.model_var = ctk.StringVar(value=default_model)
        self.model_menu = ctk.CTkOptionMenu(
            frame, variable=self.model_var, values=installed, width=140
        )
        self.model_menu.grid(row=0, column=1, padx=4, pady=10)

        # 语言选择
        ctk.CTkLabel(frame, text="语言:").grid(
            row=0, column=2, padx=(20, 4), pady=10, sticky="w"
        )

        lang_display = [f"{code} ({name})" for code, name in LANGUAGE_OPTIONS]
        self.lang_var = ctk.StringVar(value=lang_display[0])
        self.lang_menu = ctk.CTkOptionMenu(
            frame, variable=self.lang_var, values=lang_display, width=160
        )
        self.lang_menu.grid(row=0, column=3, padx=4, pady=10)

        # GPU 开关
        ctk.CTkLabel(frame, text="GPU:").grid(
            row=0, column=4, padx=(20, 4), pady=10, sticky="w"
        )
        self.gpu_var = ctk.BooleanVar(
            value=self.env.gpu_available and self.config.prefer_gpu
        )
        self.gpu_switch = ctk.CTkSwitch(
            frame, text="", variable=self.gpu_var,
            onvalue=True, offvalue=False,
        )
        self.gpu_switch.grid(row=0, column=5, padx=4, pady=10)

        if not self.env.gpu_available:
            self.gpu_switch.configure(state="disabled")
            self.gpu_var.set(False)

        self.gpu_label = ctk.CTkLabel(
            frame, text="(未检测到 GPU)" if not self.env.gpu_available else "",
            font=ctk.CTkFont(size=11), text_color="gray"
        )
        self.gpu_label.grid(row=0, column=6, padx=(2, 12), pady=10, sticky="w")

    def _build_control_section(self):
        """控制按钮 + 进度条"""
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=2, column=0, sticky="ew", padx=12, pady=4)
        frame.grid_columnconfigure(0, weight=1)

        btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
        btn_frame.grid(row=0, column=0, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶  开始转录",
            height=44, font=ctk.CTkFont(size=16, weight="bold"),
            command=self._on_start,
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 4))

        self.cancel_btn = ctk.CTkButton(
            btn_frame, text="■ 取消",
            height=44, width=100, fg_color="#c0392b", hover_color="#e74c3c",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_cancel, state="disabled",
        )
        self.cancel_btn.grid(row=0, column=1, padx=(8, 0), pady=(0, 4))

        self.progress = ctk.CTkProgressBar(frame, height=6)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(0, 2))
        self.progress.set(0)

        self.status_var = ctk.StringVar(value="就绪")
        self.status_label = ctk.CTkLabel(
            frame, textvariable=self.status_var,
            font=ctk.CTkFont(size=12), anchor="w"
        )
        self.status_label.grid(row=2, column=0, sticky="w")

    def _build_result_section(self):
        """结果文本框"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=4)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.result_box = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family="Consolas", size=13), wrap="word",
        )
        self.result_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.result_box.insert("1.0", "转录结果将显示在这里...\n")
        self.result_box.configure(state="disabled")

    def _build_bottom_section(self):
        """底部操作栏"""
        frame = ctk.CTkFrame(self)
        frame.grid(row=4, column=0, sticky="ew", padx=12, pady=(4, 12))

        self.copy_btn = ctk.CTkButton(
            frame, text="📋 复制", width=90, command=self._on_copy
        )
        self.copy_btn.pack(side="left", padx=(12, 4), pady=8)

        self.save_txt_btn = ctk.CTkButton(
            frame, text="💾 TXT", width=90, command=self._on_save_txt
        )
        self.save_txt_btn.pack(side="left", padx=4, pady=8)

        self.save_srt_btn = ctk.CTkButton(
            frame, text="🎬 SRT", width=90, command=self._on_save_srt
        )
        self.save_srt_btn.pack(side="left", padx=4, pady=8)

        self.clear_btn = ctk.CTkButton(
            frame, text="🗑 清空", width=80,
            fg_color="gray", hover_color="#555", command=self._on_clear,
        )
        self.clear_btn.pack(side="left", padx=4, pady=8)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 刷新模型列表（供外部调用）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def refresh_model_list(self):
        """刷新模型下拉框（模型管理 Tab 下载/删除后调用）"""
        installed = self.model_manager.get_installed_model_names()
        if not installed:
            installed = ["(无可用模型)"]
        self.model_menu.configure(values=installed)
        current = self.model_var.get()
        if current not in installed:
            self.model_var.set(installed[0])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 事件处理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_browse(self):
        if self.is_running:
            return
        path = filedialog.askopenfilename(
            title="选择要转录的音视频文件",
            filetypes=FILEDIALOG_TYPES,
        )
        if path:
            self.selected_file = path
            display = Path(path).name
            self.file_path_var.set(display)
            self.status_var.set(f"已选择: {display}")

    def _on_start(self):
        if self.is_running:
            return
        if not self.selected_file:
            messagebox.showwarning("提示", "请先选择一个音视频文件！")
            return
        if not self.env.ffmpeg_path:
            messagebox.showerror(
                "错误", "未找到 ffmpeg，无法处理文件！\n请将 ffmpeg.exe 放入 /bin 目录。"
            )
            return

        model_name = self.model_var.get()
        if model_name == "(无可用模型)":
            messagebox.showwarning(
                "提示", "没有可用的模型！\n请先到「模型管理」标签页下载一个模型。"
            )
            return

        self._set_running(True)
        self.all_segments.clear()
        self.audio_duration = 0.0
        self.should_cancel = False

        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.configure(state="disabled")

        self.worker_thread = threading.Thread(
            target=self._worker_main, daemon=True, name="LWF-Worker",
        )
        self.worker_thread.start()

    def _on_cancel(self):
        if self.is_running:
            self.should_cancel = True
            self.status_var.set("正在取消...")
            self.cancel_btn.configure(state="disabled")

    def _on_copy(self):
        if not self.all_segments:
            self.status_var.set("没有可复制的内容")
            return
        text = segments_to_text(self.all_segments)
        self.winfo_toplevel().clipboard_clear()
        self.winfo_toplevel().clipboard_append(text)
        self.status_var.set(f"✅ 已复制 {len(self.all_segments)} 句到剪贴板")

    def _on_save_txt(self):
        if not self.all_segments:
            messagebox.showinfo("提示", "没有可保存的内容")
            return
        default_name = Path(self.selected_file).stem + ".txt" if self.selected_file else "output.txt"
        save_path = filedialog.asksaveasfilename(
            title="保存转录文本", initialfile=default_name,
            defaultextension=".txt", filetypes=[("文本文件", "*.txt")],
        )
        if save_path:
            text = segments_to_text(self.all_segments, with_time=True)
            Path(save_path).write_text(text, encoding="utf-8")
            self.status_var.set(f"✅ 已保存: {Path(save_path).name}")

    def _on_save_srt(self):
        if not self.all_segments:
            messagebox.showinfo("提示", "没有可保存的内容")
            return
        default_name = Path(self.selected_file).stem + ".srt" if self.selected_file else "output.srt"
        save_path = filedialog.asksaveasfilename(
            title="保存 SRT 字幕", initialfile=default_name,
            defaultextension=".srt", filetypes=[("SRT 字幕", "*.srt")],
        )
        if save_path:
            srt_text = segments_to_srt(self.all_segments)
            Path(save_path).write_text(srt_text, encoding="utf-8")
            self.status_var.set(f"✅ 已保存: {Path(save_path).name}")

    def _on_clear(self):
        self.all_segments.clear()
        self.audio_duration = 0.0
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", "转录结果将显示在这里...\n")
        self.result_box.configure(state="disabled")
        self.progress.set(0)
        self.status_var.set("就绪")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 工作线程
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _worker_main(self):
        t_start = time.time()
        wav_path = None

        try:
            # Phase 1: 提取音频
            self.after(0, lambda: self.status_var.set("正在提取音频..."))
            self.after(0, lambda: self.progress.configure(mode="indeterminate"))
            self.after(0, lambda: self.progress.start())

            wav_path = extract_audio(input_path=self.selected_file, env=self.env)

            if self.should_cancel:
                self._cleanup_wav(wav_path)
                self.after(0, lambda: self._ui_on_cancelled())
                return

            # Phase 2: 流式转录
            model_name = self.model_var.get()
            model_path = self.model_manager.get_model_path_for_engine(model_name)
            device = "cuda" if self.gpu_var.get() else "cpu"
            language = self.lang_var.get()

            self.after(0, lambda: self.status_var.set(
                f"正在加载模型 [{model_name}]..."
            ))

            # 尝试转录，GPU 失败则自动回退 CPU
            seg_count = 0
            try:
                for seg in transcribe_stream(
                    wav_path=wav_path, env=self.env,
                    model_path=model_path, language=language,
                    device=device,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                ):
                    if self.should_cancel:
                        break
                    seg_count += 1
                    if seg.duration > 0 and self.audio_duration == 0:
                        self.audio_duration = seg.duration
                        self.after(0, lambda: self.progress.stop())
                        self.after(0, lambda: self.progress.configure(mode="determinate"))
                    self.after(0, lambda s=seg: self._ui_on_segment(s))

            except Exception as gpu_err:
                if device == "cuda":
                    logger.warning(f"[WORKER] GPU 转录失败，回退 CPU: {gpu_err}")
                    self.after(0, lambda: self.status_var.set(
                        f"GPU 出错，正在用 CPU 重试 [{model_name}]..."
                    ))
                    # 重置状态
                    seg_count = 0
                    self.all_segments.clear()
                    self.audio_duration = 0.0
                    self.after(0, lambda: self.progress.stop())
                    self.after(0, lambda: self.progress.configure(mode="indeterminate"))
                    self.after(0, lambda: self.progress.start())

                    for seg in transcribe_stream(
                        wav_path=wav_path, env=self.env,
                        model_path=model_path, language=language,
                        device="cpu",
                        beam_size=self.config.beam_size,
                        vad_filter=self.config.vad_filter,
                    ):
                        if self.should_cancel:
                            break
                        seg_count += 1
                        if seg.duration > 0 and self.audio_duration == 0:
                            self.audio_duration = seg.duration
                            self.after(0, lambda: self.progress.stop())
                            self.after(0, lambda: self.progress.configure(mode="determinate"))
                        self.after(0, lambda s=seg: self._ui_on_segment(s))
                else:
                    raise

            # Phase 3: 完成
            elapsed = time.time() - t_start
            self._cleanup_wav(wav_path)

            if self.should_cancel:
                self.after(0, lambda: self._ui_on_cancelled())
            else:
                self.after(0, lambda e=elapsed, c=seg_count: self._ui_on_done(c, e))

        except Exception as e:
            logger.exception("[WORKER] 转录过程出错")
            self._cleanup_wav(wav_path)
            error_msg = str(e)
            self.after(0, lambda msg=error_msg: self._ui_on_error(msg))

    def _cleanup_wav(self, wav_path):
        if wav_path:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI 回调（主线程安全）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _ui_on_segment(self, seg: SegmentInfo):
        self.all_segments.append(seg)
        line = f"[{seg.start_fmt} → {seg.end_fmt}]  {seg.text}\n"
        self.result_box.configure(state="normal")
        self.result_box.insert("end", line)
        self.result_box.see("end")
        self.result_box.configure(state="disabled")

        if self.audio_duration > 0:
            progress = min(seg.end / self.audio_duration, 1.0)
            self.progress.set(progress)

        self.status_var.set(
            f"正在转录... 第 {seg.index} 句  │  "
            f"[{seg.start_fmt} → {seg.end_fmt}]"
        )

    def _ui_on_done(self, seg_count: int, elapsed: float):
        self.progress.set(1.0)
        self.status_var.set(
            f"✅ 转录完成！共 {seg_count} 句，耗时 {elapsed:.1f} 秒"
        )
        self._set_running(False)

    def _ui_on_cancelled(self):
        count = len(self.all_segments)
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)
        self.status_var.set(f"已取消（已识别 {count} 句）")
        self._set_running(False)

    def _ui_on_error(self, error_msg: str):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)
        self.status_var.set("❌ 转录失败")
        self._set_running(False)
        messagebox.showerror("转录错误", f"转录过程中发生错误：\n\n{error_msg}")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI 状态管理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _set_running(self, running: bool):
        self.is_running = running
        if running:
            self.start_btn.configure(state="disabled", text="⏳ 转录中...")
            self.cancel_btn.configure(state="normal")
            self.browse_btn.configure(state="disabled")
            self.model_menu.configure(state="disabled")
            self.lang_menu.configure(state="disabled")
            if self.env.gpu_available:
                self.gpu_switch.configure(state="disabled")
        else:
            self.start_btn.configure(state="normal", text="▶  开始转录")
            self.cancel_btn.configure(state="disabled")
            self.browse_btn.configure(state="normal")
            self.model_menu.configure(state="normal")
            self.lang_menu.configure(state="normal")
            if self.env.gpu_available:
                self.gpu_switch.configure(state="normal")
