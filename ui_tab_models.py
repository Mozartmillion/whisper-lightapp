"""
ui_tab_models.py — 模型管理 Tab 页

功能：
  - 展示所有可用模型的状态（卡片式）
  - 下载 / 删除模型
  - 显示下载进度
"""

import threading
import logging
from tkinter import messagebox

import customtkinter as ctk

from constants import WHISPER_MODELS
from model_manager import ModelManager, ModelInfo

logger = logging.getLogger("LWF")


class ModelsTab(ctk.CTkFrame):
    """模型管理 Tab 页"""

    def __init__(self, master, model_manager: ModelManager, on_model_changed=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.model_manager = model_manager
        self.on_model_changed = on_model_changed  # 模型变更回调
        self._card_widgets: dict[str, dict] = {}
        self._downloading: str = ""  # 当前下载中的模型名

        self._build_ui()
        self.refresh()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # UI 构建
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 顶部标题栏
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))

        ctk.CTkLabel(
            header, text="📦 模型管理",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            header, text="🔄 刷新", width=80, command=self.refresh
        ).pack(side="right", padx=4)

        # 可滚动的模型列表
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        self.scroll_frame.grid_columnconfigure(0, weight=1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 刷新模型列表
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def refresh(self):
        """重新扫描并刷新所有模型卡片"""
        # 清空旧的卡片
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self._card_widgets.clear()

        # 扫描模型
        models = self.model_manager.scan_local_models()

        for idx, (name, info) in enumerate(models.items()):
            self._create_model_card(idx, info)

    def _create_model_card(self, row: int, info: ModelInfo):
        """创建单个模型卡片"""
        card = ctk.CTkFrame(self.scroll_frame)
        card.grid(row=row, column=0, sticky="ew", padx=4, pady=4)
        card.grid_columnconfigure(1, weight=1)

        # 左侧：模型名和信息
        info_frame = ctk.CTkFrame(card, fg_color="transparent")
        info_frame.grid(row=0, column=0, sticky="w", padx=12, pady=10)

        # 状态图标 + 模型名
        status_icon = "🟢" if info.installed else "⚪"
        name_label = ctk.CTkLabel(
            info_frame,
            text=f"{status_icon}  {info.name}",
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        name_label.pack(anchor="w")

        # 描述
        desc_text = f"{info.desc}  •  {info.params} 参数  •  ~{info.size_mb} MB"
        if info.installed and info.local_size_mb > 0:
            desc_text += f"  (实际 {info.local_size_mb:.0f} MB)"
        ctk.CTkLabel(
            info_frame, text=desc_text,
            font=ctk.CTkFont(size=12), text_color="gray",
        ).pack(anchor="w")

        # 推荐显存
        ctk.CTkLabel(
            info_frame,
            text=f"推荐显存: {info.vram_gb:.0f} GB",
            font=ctk.CTkFont(size=11), text_color="#888",
        ).pack(anchor="w")

        # 中间：进度条（默认隐藏）
        progress_frame = ctk.CTkFrame(card, fg_color="transparent")
        progress_frame.grid(row=0, column=1, sticky="ew", padx=8, pady=10)

        progress_bar = ctk.CTkProgressBar(progress_frame, height=6)
        progress_bar.pack(fill="x", pady=(10, 2))
        progress_bar.set(0)
        progress_bar.pack_forget()  # 默认隐藏

        progress_label = ctk.CTkLabel(
            progress_frame, text="", font=ctk.CTkFont(size=11), text_color="gray"
        )
        progress_label.pack(anchor="w")
        progress_label.pack_forget()  # 默认隐藏

        # 右侧：操作按钮
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=0, column=2, sticky="e", padx=12, pady=10)

        if info.installed:
            delete_btn = ctk.CTkButton(
                btn_frame, text="🗑 删除", width=80,
                fg_color="#c0392b", hover_color="#e74c3c",
                command=lambda n=info.name: self._on_delete(n),
            )
            delete_btn.pack(pady=2)
            action_btn = delete_btn
        else:
            download_btn = ctk.CTkButton(
                btn_frame, text="⬇ 下载", width=80,
                command=lambda n=info.name: self._on_download(n),
            )
            download_btn.pack(pady=2)
            action_btn = download_btn

        # 缓存 widget 引用
        self._card_widgets[info.name] = {
            "card": card,
            "name_label": name_label,
            "progress_bar": progress_bar,
            "progress_label": progress_label,
            "action_btn": action_btn,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 下载操作
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_download(self, model_name: str):
        if self._downloading:
            messagebox.showwarning("提示", f"正在下载 {self._downloading}，请等待完成。")
            return

        meta = WHISPER_MODELS[model_name]
        confirm = messagebox.askyesno(
            "确认下载",
            f"即将下载模型: {model_name}\n"
            f"预估大小: ~{meta['size_mb']} MB\n\n"
            f"确认开始下载？"
        )
        if not confirm:
            return

        self._downloading = model_name
        widgets = self._card_widgets.get(model_name, {})
        if widgets:
            widgets["action_btn"].configure(state="disabled", text="下载中...")
            widgets["progress_bar"].pack(fill="x", pady=(10, 2))
            widgets["progress_bar"].set(0)
            widgets["progress_label"].pack(anchor="w")
            widgets["progress_label"].configure(text="准备下载...")

        # 后台线程下载
        thread = threading.Thread(
            target=self._download_worker,
            args=(model_name,),
            daemon=True,
        )
        thread.start()

    def _download_worker(self, model_name: str):
        """后台下载线程"""
        try:
            self.model_manager.download_model(
                model_name,
                progress_callback=lambda p: self.after(
                    0, lambda p=p: self._update_download_progress(p)
                ),
            )
            self.after(0, lambda: self._on_download_complete(model_name, success=True))
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: self._on_download_complete(
                model_name, success=False, error=error_msg
            ))

    def _update_download_progress(self, progress):
        """更新下载进度 UI"""
        widgets = self._card_widgets.get(progress.model_name, {})
        if not widgets:
            return

        if progress.status == "downloading":
            widgets["progress_bar"].set(progress.progress)
            if progress.speed_mb > 0:
                widgets["progress_label"].configure(
                    text=f"{progress.downloaded_mb:.1f} / {progress.total_mb:.0f} MB  "
                         f"({progress.speed_mb:.1f} MB/s)"
                )
            else:
                widgets["progress_label"].configure(text="下载中...")

    def _on_download_complete(self, model_name: str, success: bool, error: str = ""):
        """下载完成回调"""
        self._downloading = ""
        if success:
            messagebox.showinfo("下载完成", f"模型 {model_name} 下载成功！")
        else:
            messagebox.showerror("下载失败", f"模型 {model_name} 下载失败：\n\n{error}")

        self.refresh()
        if self.on_model_changed:
            self.on_model_changed()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 删除操作
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _on_delete(self, model_name: str):
        confirm = messagebox.askyesno(
            "确认删除",
            f"确认删除模型: {model_name}？\n\n"
            f"删除后需要重新下载才能使用。"
        )
        if not confirm:
            return

        success = self.model_manager.delete_model(model_name)
        if success:
            messagebox.showinfo("删除成功", f"模型 {model_name} 已删除。")
        else:
            messagebox.showerror("删除失败", f"模型 {model_name} 删除失败。")

        self.refresh()
        if self.on_model_changed:
            self.on_model_changed()
