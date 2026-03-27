"""
core_transcribe.py — LWF 转录核心逻辑

本模块只暴露核心函数，不涉及任何 UI 逻辑：

  1. extract_audio(input_path, env) -> Path
     调用 ffmpeg 将任意音视频转为 16kHz mono WAV。

  2. transcribe_stream(wav_path, env, ...) -> Generator[SegmentInfo, ...]
     封装 faster-whisper，逐句 yield 识别进度与文本。

字幕切分策略（v2.1 语义分句升级）：
  - 主力：wtpsplit SaT 模型（sat-3l-sm），基于 AI 语义理解做分句
    ✅ 85+ 语言（英/中/日/韩等），不依赖标点
    ✅ 语义完整性最优，适合后续翻译
  - 兜底：规则切分（标点 + 连词 + 时长/字数阈值）
    用于 wtpsplit 不可用或超长句的二次处理

典型调用流程（在 UI 线程外）：
    wav = extract_audio("input.mp4", env)
    for seg in transcribe_stream(wav, env, model_name="base"):
        print(seg)
"""

import sys
import re
import subprocess
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Generator

logger = logging.getLogger("LWF")

# ═══════════════════════════════════════════════════════════════════════════
# wtpsplit 语义分句引擎（懒加载）
# ═══════════════════════════════════════════════════════════════════════════
_sat_model = None  # 全局缓存，避免重复加载 ~430MB 模型


def _get_sat_model():
    """
    懒加载 wtpsplit SaT 语义分句模型。
    使用 sat-3l-sm（3 层 Transformer，轻量高速，85+ 语言）。
    首次调用会从 HuggingFace 自动下载 ONNX 模型。
    """
    global _sat_model
    if _sat_model is not None:
        return _sat_model
    try:
        from wtpsplit_lite import SaT
        logger.info("[CORE] 正在加载 wtpsplit SaT 语义分句模型 (sat-3l-sm)...")
        t0 = time.time()
        _sat_model = SaT("sat-3l-sm")
        logger.info(f"[CORE] SaT 模型加载完成，耗时 {time.time() - t0:.1f}s")
        return _sat_model
    except ImportError:
        logger.warning("[CORE] wtpsplit-lite 未安装，将使用规则切分作为兜底")
        return None
    except Exception as e:
        logger.warning(f"[CORE] SaT 模型加载失败，使用规则切分: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SegmentInfo:
    """
    单句转录结果。每 yield 一次就是一句。
    UI 侧拿到后可直接拼接到文本框。
    """
    index: int           # 句子序号，从 1 开始
    start: float         # 起始时间（秒）
    end: float           # 结束时间（秒）
    text: str            # 识别出的文本
    # —— 以下字段在第一句时一并提供元信息 ——
    language: str = ""           # 检测到的语言代码
    language_prob: float = 0.0   # 语言检测置信度
    duration: float = 0.0        # 音频总时长（秒）

    @property
    def start_fmt(self) -> str:
        """格式化起始时间 HH:MM:SS"""
        return _fmt_time(self.start)

    @property
    def end_fmt(self) -> str:
        """格式化结束时间 HH:MM:SS"""
        return _fmt_time(self.end)

    @property
    def timestamp_srt(self) -> str:
        """SRT 格式时间轴"""
        return f"{_fmt_srt(self.start)} --> {_fmt_srt(self.end)}"


# ═══════════════════════════════════════════════════════════════════════════
# 函数 1：extract_audio
# ═══════════════════════════════════════════════════════════════════════════

def extract_audio(
    input_path: str,
    env,                        # utils_env.EnvInfo
    sample_rate: int = 16000,
    output_filename: Optional[str] = None,
) -> Path:
    """
    使用 ffmpeg 将任意音视频文件转为 16kHz 单声道 WAV。
    """
    input_file = Path(input_path).resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 确定 ffmpeg 路径
    if env.ffmpeg_path and env.ffmpeg_path.is_file():
        ffmpeg_exe = str(env.ffmpeg_path)
    else:
        ffmpeg_exe = "ffmpeg"
        logger.warning("[CORE] env.ffmpeg_path 不可用，回退到系统 PATH 中的 ffmpeg")

    # 输出路径
    if output_filename:
        wav_file = env.temp_dir / output_filename
    else:
        safe_stem = re.sub(r'[^\w\-.]', '_', input_file.stem)
        wav_file = env.temp_dir / f"{safe_stem}_temp.wav"

    wav_file.parent.mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = [
        ffmpeg_exe,
        "-hide_banner", "-y",
        "-i", str(input_file),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(wav_file),
    ]

    logger.info(f"[CORE] ffmpeg 命令: {' '.join(cmd)}")

    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    t0 = time.time()
    timeout_seconds = 600

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        _, stderr_bytes = proc.communicate(timeout=timeout_seconds)
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"ffmpeg 超时（>{timeout_seconds}s），文件可能过大: {input_file.name}"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "找不到 ffmpeg 可执行文件！\n"
            "请确保 ffmpeg.exe 已放入 /bin 目录，或已安装到系统 PATH 中。"
        )

    elapsed = time.time() - t0

    if returncode != 0:
        err_lines = stderr_text.strip().split("\n")
        err_tail = "\n".join(err_lines[-5:]) if len(err_lines) > 5 else stderr_text.strip()
        raise RuntimeError(
            f"ffmpeg 退出码 {returncode}，文件: {input_file.name}\n"
            f"错误信息:\n{err_tail}"
        )

    if not wav_file.exists():
        raise RuntimeError(f"ffmpeg 执行完毕但输出文件不存在: {wav_file}")

    file_size = wav_file.stat().st_size
    if file_size == 0:
        wav_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg 输出文件为空（0 字节），输入文件可能没有音轨: {input_file.name}"
        )

    logger.info(
        f"[CORE] 音频提取完成: {wav_file.name} "
        f"({file_size / 1024 / 1024:.1f} MB, 耗时 {elapsed:.1f}s)"
    )
    return wav_file


# ═══════════════════════════════════════════════════════════════════════════
# 模型缓存（避免频繁创建/销毁导致 CUDA 资源释放崩溃）
# ═══════════════════════════════════════════════════════════════════════════
_model_cache: dict = {
    "model": None,        # WhisperModel 实例
    "model_path": None,   # 当前加载的模型路径
    "device": None,       # 当前设备
    "compute_type": None, # 当前计算类型
}


def _get_or_load_model(model_path: str, device: str, compute_type: str, download_root: str):
    """
    获取缓存的模型，如果参数变化则重新加载。
    缓存的目的是避免模型对象被 GC 析构时触发 CUDA 资源释放崩溃。
    """
    from faster_whisper import WhisperModel

    cache = _model_cache
    if (cache["model"] is not None
        and cache["model_path"] == model_path
        and cache["device"] == device
        and cache["compute_type"] == compute_type):
        logger.info("[CORE] 使用缓存的模型实例")
        return cache["model"]

    logger.info(f"[CORE] 加载模型: {model_path} ({device}/{compute_type})")
    t0 = time.time()

    model = WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        download_root=download_root,
    )

    # 更新缓存
    cache["model"] = model
    cache["model_path"] = model_path
    cache["device"] = device
    cache["compute_type"] = compute_type

    logger.info(f"[CORE] 模型加载完成，耗时 {time.time() - t0:.1f}s")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 函数 2：transcribe_stream
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_stream(
    wav_path: str | Path,
    env,                             # utils_env.EnvInfo
    model_path: str = "base",        # 本地路径或模型名
    language: Optional[str] = None,
    device: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    vad_min_silence_ms: int = 500,
) -> Generator[SegmentInfo, None, None]:
    """
    封装 faster-whisper 引擎，逐句 yield 识别结果。
    """
    wav_path = Path(wav_path).resolve()
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV 文件不存在: {wav_path}")

    # 设备决策
    if device == "auto":
        actual_device = "cuda" if env.gpu_available else "cpu"
    else:
        actual_device = device

    compute_type = "float16" if actual_device == "cuda" else "int8"

    logger.info(
        f"[CORE] 准备转录: model={model_path}, device={actual_device}, "
        f"compute={compute_type}, beam={beam_size}, vad={vad_filter}"
    )

    # 加载模型（使用缓存）
    try:
        model = _get_or_load_model(
            model_path, actual_device, compute_type, str(env.models_dir)
        )
    except Exception as e:
        error_msg = str(e)
        if "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
            logger.warning(f"[CORE] GPU 加载失败，自动回退 CPU 模式: {e}")
            actual_device = "cpu"
            compute_type = "int8"
            model = _get_or_load_model(
                model_path, "cpu", "int8", str(env.models_dir)
            )
        else:
            raise RuntimeError(f"模型加载失败 [{model_path}]: {e}") from e

    # 处理语言参数
    lang_code = None
    if language and language.lower() not in ("auto", "auto (自动检测)"):
        lang_code = language.split(" ")[0].split("(")[0].strip()

    # VAD 参数
    vad_params = None
    if vad_filter:
        vad_params = dict(min_silence_duration_ms=vad_min_silence_ms)

    # 执行转录（启用词级时间戳 + 句子切分）
    t_trans_start = time.time()

    segments_iter, info = model.transcribe(
        str(wav_path),
        language=lang_code,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        word_timestamps=True,         # 启用词级时间戳，用于精确切分
    )

    detected_lang = info.language
    detected_prob = info.language_probability
    audio_duration = info.duration

    logger.info(
        f"[CORE] 转录开始: 语言={detected_lang} (置信度={detected_prob:.2%}), "
        f"音频时长={audio_duration:.1f}s"
    )

    # 逐句 yield（对长 segment 做二次切分 + 跨 segment 短片段合并）
    idx = 0
    # 缓冲区：持有上一个片段，等下一个来了才决定是否合并
    buffer: tuple[float, float, str] | None = None

    # 获取语言感知的参数配置
    _profile = _get_lang_profile(detected_lang)
    _buf_joiner = _profile["joiner"]
    _buf_min_chars = _profile["min_chars"]
    _buf_min_duration = _profile["min_duration"]
    _buf_force_split_duration = _profile["force_split_duration"]
    _buf_force_split_chars = _profile["force_split_chars"]

    def _is_short(start, end, text):
        """判断片段是否过短，需要合并（使用语言感知的参数）"""
        return (len(text) < _buf_min_chars
                or (end - start) < _buf_min_duration)

    def _yield_seg(start, end, text):
        """输出一个最终的 SegmentInfo"""
        nonlocal idx
        text = text.strip()
        if not text:
            return None
        idx += 1
        seg_info = SegmentInfo(index=idx, start=start, end=end, text=text)
        if idx == 1:
            seg_info.language = detected_lang
            seg_info.language_prob = detected_prob
            seg_info.duration = audio_duration
        logger.debug(f"[CORE] #{idx} [{seg_info.start_fmt} -> {seg_info.end_fmt}] {text}")
        return seg_info

    for segment in segments_iter:
        sub_segments = _split_segment_by_sentences(segment, lang=detected_lang)

        for start, end, text in sub_segments:
            text = text.strip()
            if not text:
                continue

            if buffer is None:
                # 缓冲区为空，先存入
                buffer = (start, end, text)
            elif _is_short(*buffer):
                # 缓冲区里的片段太短 → 尝试合并到当前片段
                buf_start, buf_end, buf_text = buffer
                merged_duration = end - buf_start
                merged_chars = len(buf_text) + len(text)
                if merged_duration <= _buf_force_split_duration and merged_chars <= _buf_force_split_chars:
                    buffer = (buf_start, end, buf_text + _buf_joiner + text)
                else:
                    # 合并后超限，强制输出缓冲区
                    seg = _yield_seg(*buffer)
                    if seg:
                        yield seg
                    buffer = (start, end, text)
            elif _is_short(start, end, text):
                # 当前片段太短 → 尝试合并到缓冲区
                buf_start, buf_end, buf_text = buffer
                merged_duration = end - buf_start
                merged_chars = len(buf_text) + len(text)
                if merged_duration <= _buf_force_split_duration and merged_chars <= _buf_force_split_chars:
                    buffer = (buf_start, end, buf_text + _buf_joiner + text)
                else:
                    # 合并后超限，先输出缓冲区，短片段存入新缓冲
                    seg = _yield_seg(*buffer)
                    if seg:
                        yield seg
                    buffer = (start, end, text)
            else:
                # 两个都不短 → 输出缓冲区，当前存入缓冲
                seg = _yield_seg(*buffer)
                if seg:
                    yield seg
                buffer = (start, end, text)

    # 输出最后一个缓冲的片段
    if buffer:
        seg = _yield_seg(*buffer)
        if seg:
            yield seg

    t_trans_elapsed = time.time() - t_trans_start
    if t_trans_elapsed > 0:
        logger.info(
            f"[CORE] 转录完成: {idx} 句, 音频 {audio_duration:.1f}s, "
            f"转录耗时 {t_trans_elapsed:.1f}s "
            f"(速比 {audio_duration / t_trans_elapsed:.1f}x)"
        )
    else:
        logger.info(f"[CORE] 转录完成: {idx} 句")


# ═══════════════════════════════════════════════════════════════════════════
# 句子切分（语义分句 + 规则兜底）
# ═══════════════════════════════════════════════════════════════════════════

# ---------- 语言感知的字幕参数 ----------
# CJK 语言（日/中/韩）: 字符信息密度高，每字≈1语义单位
# 拉丁语言（英/法/德等）: 字符信息密度低，每字只是字母
_LANG_PROFILES = {
    "cjk": {  # 日语、中文、韩语
        "max_chars_per_subtitle": 45,   # CJK 字符信息密度高，45字是显示上限
        "max_duration": 8.0,            # wtpsplit 句超过此时长就二次规则切分
        "force_split_duration": 6.0,    # 规则切分时单句最大时长
        "force_split_chars": 45,        # 规则切分时单句最大字符数（放宽到45减少劈词）
        "min_chars": 3,                 # 3个CJK字符就有语义
        "min_duration": 0.8,
        "joiner": "",                   # CJK 不需要空格拼接
        "clause_split_duration": 3.0,   # 次级标点处断句的最小时长
        "clause_split_chars": 25,       # 次级标点处断句的最小字符数
        "min_chunk_chars": 12,          # 强制切分时每段最少字符（缩小以留更多回退空间）
    },
    "latin": {  # 英语、法语、德语、西班牙语等
        "max_chars_per_subtitle": 80,   # 英语 80 字符 ≈ 12-15 单词
        "max_duration": 8.0,
        "force_split_duration": 6.0,
        "force_split_chars": 80,
        "min_chars": 10,                # 英语 10 字符 ≈ 2 个短单词
        "min_duration": 0.8,
        "joiner": " ",                  # 拉丁语系需要空格拼接
        "clause_split_duration": 3.0,
        "clause_split_chars": 40,
        "min_chunk_chars": 30,
    },
}

# 语言代码 → profile 映射
_CJK_LANGS = {"ja", "zh", "ko", "yue"}   # 日语、中文、韩语、粤语

def _get_lang_profile(lang: str | None) -> dict:
    """根据语言代码获取对应的切分参数配置"""
    if lang and lang.lower() in _CJK_LANGS:
        return _LANG_PROFILES["cjk"]
    return _LANG_PROFILES["latin"]

# ---------- 规则切分的标点/连词 ----------
_SENTENCE_ENDINGS = {'.', '!', '?', '。', '！', '？', '；', '…'}
_CLAUSE_ENDINGS = {',', '，', ':', '：', ';', '、'}
_BREAK_BEFORE_WORDS = {
    'and', 'but', 'or', 'so', 'because', 'when', 'where', 'which',
    'that', 'if', 'while', 'although', 'though', 'since', 'unless',
    'before', 'after', 'like', 'as', 'than',
}

# ---------- 日语助词断点（CJK 劈词防护）----------
# 这些平假名助词/助动词后面通常是词边界，适合作为断句点
# は(主题), が(主格), を(宾格), に(目标), で(手段/场所),
# と(并列/引用), も(也), の(所属), へ(方向), や(列举), か(疑问)
_JA_PARTICLES_FOR_SPLIT = set('はがをにでともへやか')

def _is_hiragana_char(ch: str) -> bool:
    """判断单个字符是否为平假名"""
    return 0x3040 <= ord(ch) <= 0x309F

# ---------- 兼容旧代码的默认值（fallback 用） ----------
_MIN_CHARS = 5
_MIN_WORDS = 2
_MIN_DURATION = 0.8
_MAX_DURATION = 8.0
_FORCE_SPLIT_DURATION = 6.0
_FORCE_SPLIT_CHARS = 80


def _split_segment_by_sentences(segment, lang: str | None = None) -> list[tuple[float, float, str]]:
    """
    将一个 faster-whisper segment 切分为适合字幕的短句。

    优先使用 wtpsplit SaT 语义分句（AI 模型，85+ 语言，语义完整性最优），
    如果 wtpsplit 不可用则回退到规则切分。

    参数:
        segment: faster-whisper segment 对象
        lang: 语言代码（如 'ja', 'en', 'zh'），用于语言感知的参数调整
    """
    words = segment.words if hasattr(segment, 'words') and segment.words else None

    if not words:
        return [(segment.start, segment.end, segment.text.strip())]

    profile = _get_lang_profile(lang)

    # 尝试语义分句
    sat = _get_sat_model()
    if sat is not None:
        try:
            return _split_with_wtpsplit(words, sat, profile=profile)
        except Exception as e:
            logger.warning(f"[CORE] wtpsplit 分句失败，回退规则切分: {e}")

    # 规则切分（fallback）
    return _split_with_rules(words, segment, profile=profile)


def _split_with_wtpsplit(words, sat, profile: dict | None = None) -> list[tuple[float, float, str]]:
    """
    使用 wtpsplit SaT 模型做语义分句，再用词级时间戳反算时间轴。

    v2.1.1 修复版 — 改用词索引直接映射，彻底解决日语等无空格语言的映射错乱问题。
    v2.2.0 — 支持语言感知参数（profile）。
    """
    if profile is None:
        profile = _LANG_PROFILES["latin"]

    p_max_duration = profile["max_duration"]
    p_force_split_chars = profile["force_split_chars"]
    p_min_chunk_chars = profile["min_chunk_chars"]
    # ── Step 1: 构建拼接文本 + 词的字符区间索引 ──
    # word_spans[i] = (start_char, end_char) 表示第 i 个词在 full_text 中的字符区间 [start, end)
    word_spans: list[tuple[int, int]] = []
    full_text_parts: list[str] = []
    pos = 0

    for w in words:
        raw_word = w.word                    # whisper 原始词文本（可能含前导空格）
        wt = raw_word.strip()
        if not wt:
            word_spans.append((pos, pos))    # 空词，零长度区间
            continue
        # 保留 whisper 原始的前导空格作为词间分隔
        # 这样日语无空格时自然拼接，英语有空格时也正确
        has_leading_space = raw_word.startswith(" ") or raw_word.startswith("\u3000")
        if pos > 0 and has_leading_space:
            full_text_parts.append(" ")
            pos += 1
        word_spans.append((pos, pos + len(wt)))
        full_text_parts.append(wt)
        pos += len(wt)

    full_text = "".join(full_text_parts)

    if not full_text.strip():
        return [(words[0].start, words[-1].end, "")]

    # ── Step 2: wtpsplit 语义分句 ──
    sentences = sat.split(full_text)

    if not sentences:
        return [(words[0].start, words[-1].end, full_text)]

    logger.debug(f"[CORE] wtpsplit 分出 {len(sentences)} 句 (从 {len(words)} 个词)")

    # ── Step 3: 将语义句映射回词级时间戳 ──
    # 核心思路：逐句在 full_text 中定位字符区间，然后反查落在该区间内的词索引
    results = []
    search_pos = 0

    for sent_text in sentences:
        sent_clean = sent_text.strip()
        if not sent_clean:
            continue

        # 在 full_text 中定位这个句子
        sent_start = full_text.find(sent_clean, search_pos)
        if sent_start == -1:
            # 尝试模糊匹配
            sent_start = _fuzzy_find(full_text, sent_clean, search_pos)

        if sent_start == -1:
            # 最后手段：从头开始找（不限制 search_pos）
            sent_start = full_text.find(sent_clean)
            if sent_start == -1:
                logger.warning(f"[CORE] wtpsplit 句子映射失败，跳过: {sent_clean[:30]}...")
                continue

        sent_end = sent_start + len(sent_clean)  # exclusive

        # 反查词索引：找所有字符区间与 [sent_start, sent_end) 有重叠的词
        first_word_idx = None
        last_word_idx = None
        for wi, (ws, we) in enumerate(word_spans):
            if ws == we:
                continue  # 跳过空词
            # 检查词区间 [ws, we) 是否与句子区间 [sent_start, sent_end) 有重叠
            if we > sent_start and ws < sent_end:
                if first_word_idx is None:
                    first_word_idx = wi
                last_word_idx = wi

        if first_word_idx is not None and last_word_idx is not None:
            start_time = words[first_word_idx].start
            end_time = words[last_word_idx].end
            results.append((start_time, end_time, sent_clean))
        else:
            logger.warning(f"[CORE] 词索引映射失败: {sent_clean[:30]}...")

        # 推进搜索位置
        search_pos = sent_end

    # 如果映射全部失败，保底输出整个 segment
    if not results:
        logger.warning("[CORE] wtpsplit 映射全部失败，回退为整段输出")
        return [(words[0].start, words[-1].end, full_text)]

    # ── Step 4: 对超长句做二次规则切分（字幕显示限制） ──
    # 同时强制执行 max_duration 限制
    final_results = []
    for start, end, text in results:
        duration = end - start
        if duration > p_max_duration or len(text) > p_force_split_chars:
            # 收集这段时间范围内的词
            sub_words = [w for w in words
                         if w.start >= start - 0.05 and w.end <= end + 0.05
                         and w.word.strip()]
            if len(sub_words) >= 2:
                sub_results = _rule_split_words(sub_words, profile=profile)
                final_results.extend(sub_results)
            else:
                final_results.append((start, end, text))
        else:
            final_results.append((start, end, text))

    # ── Step 5: 合并过短片段 ──
    final_results = _merge_short_segments(final_results, profile=profile)

    # ── Step 6: 纯字符数强制切分兜底 ──
    final_final = []
    for start, end, text in final_results:
        if len(text) > p_force_split_chars:
            chunks = _force_split_by_chars(text, start, end, profile=profile)
            final_final.extend(chunks)
        else:
            final_final.append((start, end, text))

    return final_final if final_final else [(words[0].start, words[-1].end, full_text)]



def _force_split_by_chars(
    text: str, start: float, end: float, profile: dict | None = None
) -> list[tuple[float, float, str]]:
    """
    纯字符数强制切分（最终兜底）。

    当 Whisper 的 word timestamps 全部被压缩到极短时间范围（1-2 秒）时，
    基于时间戳的切分和合并逻辑都会失效。这个函数不看时间戳，
    纯按字符数在标点处切分，时间按字符比例均匀分配。

    v2.2.1 — CJK 助词感知：在没有标点可断时，优先在日语助词/助动词后断开，
    避免把「追加」「専用」「向かっている」等词从中间劈开。
    """
    if profile is None:
        profile = _LANG_PROFILES["latin"]

    max_chars = profile["force_split_chars"]
    min_chunk = profile["min_chunk_chars"]

    if len(text) <= max_chars:
        return [(start, end, text)]

    duration = end - start
    total_chars = len(text)

    # 优先在标点处切分
    _SPLIT_POINTS = set('。！？、，,. ）」』】》!?')
    _GOOD_SPLIT_POINTS = set('。！？.!?')

    # ── CJK 助词断点 ──
    # 日语助词/助动词后面是天然的词边界，在此断开不会劈开词语
    # 格式：在 text[p] 为以下字符 **且后面接的是汉字/片假名** 时，p+1 是安全断点
    _JA_PARTICLES = set('はがをにでとものへやかもらなりたてけ')
    # 假名范围：平假名 U+3040-309F, 片假名 U+30A0-30FF, CJK U+4E00-9FFF
    def _is_cjk_or_katakana(ch: str) -> bool:
        cp = ord(ch)
        return (0x4E00 <= cp <= 0x9FFF      # CJK 统一汉字
                or 0x30A0 <= cp <= 0x30FF    # 片假名
                or 0x3400 <= cp <= 0x4DBF    # CJK 扩展 A
                or 0xF900 <= cp <= 0xFAFF)   # CJK 兼容汉字

    def _is_hiragana(ch: str) -> bool:
        return 0x3040 <= ord(ch) <= 0x309F

    def _is_good_particle_break(pos: int) -> bool:
        """判断 text[pos] 后面（pos+1）是否为助词后的安全断点。
        条件：text[pos] 是助词 + 后面一个字符是汉字/片假名（新词的开头）"""
        if pos + 1 >= total_chars:
            return False
        ch = text[pos]
        next_ch = text[pos + 1]
        if ch not in _JA_PARTICLES:
            return False
        # 助词必须是平假名（排除汉字「は」=「刃」等误判）
        if not _is_hiragana(ch):
            return False
        # 后面必须接汉字或片假名（表示新词开始）
        return _is_cjk_or_katakana(next_ch)

    chunks = []
    chunk_start_char = 0

    i = 0
    while i < total_chars:
        remaining = total_chars - chunk_start_char
        if remaining <= max_chars:
            # 剩余部分不超限，直接收尾
            break

        # 在 [chunk_start + min_chunk, chunk_start + max_chars] 范围内找最佳切分点
        search_start = chunk_start_char + min_chunk
        search_end = min(chunk_start_char + max_chars, total_chars)

        best_pos = -1
        # 优先级1: 句号类标点
        for p in range(search_end - 1, search_start - 1, -1):
            if text[p] in _GOOD_SPLIT_POINTS:
                best_pos = p + 1
                break
        # 优先级2: 逗号类标点
        if best_pos == -1:
            for p in range(search_end - 1, search_start - 1, -1):
                if text[p] in _SPLIT_POINTS:
                    best_pos = p + 1
                    break
        # 优先级3: 日语助词后断开（避免劈词）
        if best_pos == -1:
            for p in range(search_end - 1, search_start - 1, -1):
                if _is_good_particle_break(p):
                    best_pos = p + 1  # 在助词后面断
                    break
        # 优先级4: 都找不到就硬切
        if best_pos == -1:
            best_pos = search_end

        chunk_text = text[chunk_start_char:best_pos].strip()
        if chunk_text:
            # 按字符比例分配时间
            ratio_start = chunk_start_char / total_chars
            ratio_end = best_pos / total_chars
            t_start = start + duration * ratio_start
            t_end = start + duration * ratio_end
            chunks.append((t_start, t_end, chunk_text))

        chunk_start_char = best_pos
        i = best_pos

    # 处理最后一段
    if chunk_start_char < total_chars:
        last_text = text[chunk_start_char:].strip()
        if last_text:
            ratio_start = chunk_start_char / total_chars
            t_start = start + duration * ratio_start
            chunks.append((t_start, end, last_text))

    return chunks if chunks else [(start, end, text)]


def _fuzzy_find(haystack: str, needle: str, start: int = 0) -> int:
    """模糊查找：忽略多余空格的差异"""
    # 先尝试直接找
    pos = haystack.find(needle, start)
    if pos != -1:
        return pos

    # 压缩空格后查找
    needle_compact = re.sub(r'\s+', ' ', needle).strip()
    compact_text = ""
    compact_to_orig: list[int] = []
    for i in range(start, len(haystack)):
        ch = haystack[i]
        if ch in (' ', '\t', '\n'):
            if compact_text and compact_text[-1] != ' ':
                compact_text += ' '
                compact_to_orig.append(i)
        else:
            compact_text += ch
            compact_to_orig.append(i)

    pos = compact_text.find(needle_compact)
    if pos != -1 and pos < len(compact_to_orig):
        return compact_to_orig[pos]

    return -1


def _rule_split_words(words_list, profile: dict | None = None) -> list[tuple[float, float, str]]:
    """
    纯规则切分一组词（用于对 wtpsplit 产出的超长句做二次切分）。
    v2.2.0 — 支持语言感知参数（profile）。
    """
    if not words_list:
        return []

    if profile is None:
        profile = _LANG_PROFILES["latin"]

    p_force_split_duration = profile["force_split_duration"]
    p_force_split_chars = profile["force_split_chars"]
    p_clause_split_duration = profile["clause_split_duration"]
    p_clause_split_chars = profile["clause_split_chars"]

    results = []
    current_words = []
    current_start = words_list[0].start
    current_chars = 0

    def _flush():
        nonlocal current_words, current_start, current_chars
        if not current_words:
            return
        text = "".join(w.word for w in current_words).strip()
        if text:
            results.append((current_start, current_words[-1].end, text))
        current_words = []
        current_chars = 0

    for i, word in enumerate(words_list):
        word_text = word.word.strip()
        word_char_len = len(word.word)

        current_words.append(word)
        current_chars += word_char_len
        current_duration = word.end - current_start

        should_split = False

        # 规则 1: 句子结束标点
        if word_text and word_text[-1] in _SENTENCE_ENDINGS:
            should_split = True
        # 规则 2: 次级标点 + 超阈值（使用语言感知的参数）
        elif word_text and word_text[-1] in _CLAUSE_ENDINGS:
            if current_duration >= p_clause_split_duration or current_chars >= p_clause_split_chars:
                should_split = True
        # 规则 3: 连词前断（主要对英语有效）
        elif (i + 1 < len(words_list)
              and current_duration >= p_clause_split_duration
              and words_list[i + 1].word.strip().lower() in _BREAK_BEFORE_WORDS
              and current_chars >= p_clause_split_chars):
            should_split = True
        # 规则 3.5: 日语助词后断句（CJK 专用）
        # 当词以助词结尾（如「は」「が」「を」「に」「で」「と」「も」）
        # 且积累字符数超过 clause_split_chars 时，在助词后断开
        elif (word_text
              and len(word_text) >= 1
              and word_text[-1] in _JA_PARTICLES_FOR_SPLIT
              and _is_hiragana_char(word_text[-1])
              and current_chars >= p_clause_split_chars):
            should_split = True
        # 规则 4: 硬限制
        elif current_duration >= p_force_split_duration or current_chars >= p_force_split_chars:
            should_split = True

        if should_split:
            _flush()
            if i + 1 < len(words_list):
                current_start = words_list[i + 1].start

    _flush()
    return results


def _split_with_rules(words, segment, profile: dict | None = None) -> list[tuple[float, float, str]]:
    """
    纯规则切分（wtpsplit 不可用时的 fallback）。
    v2.2.0 — 支持语言感知参数（profile）。
    """
    if profile is None:
        profile = _LANG_PROFILES["latin"]

    p_force_split_chars = profile["force_split_chars"]

    results = _rule_split_words(words, profile=profile)
    results = _merge_short_segments(results, profile=profile)
    # 兜底：强制切分超长条
    final = []
    for start, end, text in results:
        if len(text) > p_force_split_chars:
            final.extend(_force_split_by_chars(text, start, end, profile=profile))
        else:
            final.append((start, end, text))
    return final if final else [(segment.start, segment.end, segment.text.strip())]


def _merge_short_segments(
    segments: list[tuple[float, float, str]],
    profile: dict | None = None,
) -> list[tuple[float, float, str]]:
    """
    合并过短的字幕片段到相邻片段。

    规则：如果一个片段太短（字符数 < min_chars 或时长 < min_duration），
    将它合并到时间上更近的相邻片段（优先合并到后一个）。
    合并后时长不得超过 max_duration **且** 字符数不得超过 force_split_chars，
    防止过度合并。
    v2.2.0 — 支持语言感知参数（profile）。
    """
    if len(segments) <= 1:
        return segments

    if profile is None:
        profile = _LANG_PROFILES["latin"]

    p_min_chars = profile["min_chars"]
    p_min_duration = profile["min_duration"]
    p_max_duration = profile["max_duration"]
    p_force_split_chars = profile["force_split_chars"]
    p_joiner = profile["joiner"]

    merged = list(segments)
    changed = True

    def _can_merge(text_a: str, text_b: str, start: float, end: float) -> bool:
        """检查合并后是否在允许范围内（时长 + 字符数双重限制）"""
        return ((end - start) <= p_max_duration
                and (len(text_a) + len(text_b)) <= p_force_split_chars)

    while changed:
        changed = False
        i = 0
        while i < len(merged):
            start, end, text = merged[i]
            char_count = len(text)
            duration = end - start

            is_short = (char_count < p_min_chars
                        or duration < p_min_duration)

            if is_short and len(merged) > 1:
                if i + 1 < len(merged):
                    # 尝试合并到后一个片段
                    next_start, next_end, next_text = merged[i + 1]
                    if _can_merge(text, next_text, start, next_end):
                        merged[i + 1] = (start, next_end, text + p_joiner + next_text)
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1  # 合并后会超限，跳过
                elif i > 0:
                    # 最后一个短片段，尝试合并到前一个
                    prev_start, prev_end, prev_text = merged[i - 1]
                    if _can_merge(prev_text, text, prev_start, end):
                        merged[i - 1] = (prev_start, end, prev_text + p_joiner + text)
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1  # 合并后会超限，跳过
                else:
                    i += 1
            else:
                i += 1

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_time(seconds: float) -> str:
    """秒 → HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def _fmt_srt(seconds: float) -> str:
    """秒 → SRT 时间戳 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[SegmentInfo]) -> str:
    """将 SegmentInfo 列表转为 SRT 字幕字符串"""
    lines = []
    for seg in segments:
        lines.append(str(seg.index))
        lines.append(seg.timestamp_srt)
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def segments_to_text(segments: list[SegmentInfo], with_time: bool = False) -> str:
    """将 SegmentInfo 列表转为纯文本"""
    if with_time:
        return "\n".join(
            f"[{seg.start_fmt} → {seg.end_fmt}]  {seg.text}"
            for seg in segments
        )
    else:
        return "\n".join(seg.text for seg in segments)
