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

    def _is_short(start, end, text):
        """判断片段是否过短，需要合并（不再用词数判断，避免日语/中文误合并）"""
        return (len(text) < _MIN_CHARS
                or (end - start) < _MIN_DURATION)

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
        sub_segments = _split_segment_by_sentences(segment)

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
                if merged_duration <= _FORCE_SPLIT_DURATION and merged_chars <= _FORCE_SPLIT_CHARS:
                    buffer = (buf_start, end, buf_text + text)
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
                if merged_duration <= _FORCE_SPLIT_DURATION and merged_chars <= _FORCE_SPLIT_CHARS:
                    buffer = (buf_start, end, buf_text + text)
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

# ---------- 字幕参数 ----------
_MAX_CHARS_PER_SUBTITLE = 80     # 每屏最大字符数（约 2 行 x 40 字符）
_MAX_DURATION = 8.0              # wtpsplit 句超过此时长就二次规则切分（秒）
_FORCE_SPLIT_DURATION = 6.0      # 规则切分时单句最大时长（秒）
_FORCE_SPLIT_CHARS = 80          # 规则切分时单句最大字符数

# ---------- 规则切分的标点/连词 ----------
_SENTENCE_ENDINGS = {'.', '!', '?', '。', '！', '？', '；', '…'}
_CLAUSE_ENDINGS = {',', '，', ':', '：', ';'}
_BREAK_BEFORE_WORDS = {
    'and', 'but', 'or', 'so', 'because', 'when', 'where', 'which',
    'that', 'if', 'while', 'although', 'though', 'since', 'unless',
    'before', 'after', 'like', 'as', 'than',
}

# ---------- 过短片段的判定阈值 ----------
_MIN_CHARS = 5          # 少于 5 个字符的片段需要合并
_MIN_WORDS = 2          # 少于 2 个词的片段需要合并
_MIN_DURATION = 0.8     # 少于 0.8 秒的片段需要合并


def _split_segment_by_sentences(segment) -> list[tuple[float, float, str]]:
    """
    将一个 faster-whisper segment 切分为适合字幕的短句。

    优先使用 wtpsplit SaT 语义分句（AI 模型，85+ 语言，语义完整性最优），
    如果 wtpsplit 不可用则回退到规则切分。

    核心流程（wtpsplit 路径）：
    1. 从 segment 提取所有词级时间戳
    2. 拼接完整文本 → 交给 wtpsplit 做语义分句
    3. 用词级时间戳反算每句的起止时间
    4. 对超长句做二次规则切分（字幕长度兜底）
    5. 合并过短片段
    """
    words = segment.words if hasattr(segment, 'words') and segment.words else None

    if not words:
        return [(segment.start, segment.end, segment.text.strip())]

    # 尝试语义分句
    sat = _get_sat_model()
    if sat is not None:
        try:
            return _split_with_wtpsplit(words, sat)
        except Exception as e:
            logger.warning(f"[CORE] wtpsplit 分句失败，回退规则切分: {e}")

    # 规则切分（fallback）
    return _split_with_rules(words, segment)


def _split_with_wtpsplit(words, sat) -> list[tuple[float, float, str]]:
    """
    使用 wtpsplit SaT 模型做语义分句，再用词级时间戳反算时间轴。

    v2.1.1 修复版 — 改用词索引直接映射，彻底解决日语等无空格语言的映射错乱问题。

    步骤：
    1. 拼接所有词的 stripped 文本（去掉 whisper 返回的前导空格），
       用分隔符标记词边界，同时记录每个词在拼接文本中的字符区间
    2. wtpsplit 对拼接文本做语义分句
    3. 根据每个句子在拼接文本中的字符位置，反查词索引区间 → 精确取时间戳
    4. 对超过 _MAX_DURATION 的句子做二次规则切分（字幕显示限制）
    5. 合并过短片段
    """
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
    # 同时强制执行 _MAX_DURATION（5秒）限制
    final_results = []
    for start, end, text in results:
        duration = end - start
        if duration > _MAX_DURATION or len(text) > _FORCE_SPLIT_CHARS:
            # 收集这段时间范围内的词
            sub_words = [w for w in words
                         if w.start >= start - 0.05 and w.end <= end + 0.05
                         and w.word.strip()]
            if len(sub_words) >= 2:
                sub_results = _rule_split_words(sub_words)
                final_results.extend(sub_results)
            else:
                final_results.append((start, end, text))
        else:
            final_results.append((start, end, text))

    # ── Step 5: 合并过短片段 ──
    final_results = _merge_short_segments(final_results)

    # ── Step 6: 纯字符数强制切分兜底 ──
    # 解决 Whisper word timestamps 压缩（所有词挤在 1-2 秒）导致 Step 4 切分后
    # 又被 Step 5 合并回去的问题。不看时间戳，纯按字符数切分，时间按比例分配。
    final_final = []
    for start, end, text in final_results:
        if len(text) > _FORCE_SPLIT_CHARS:
            chunks = _force_split_by_chars(text, start, end)
            final_final.extend(chunks)
        else:
            final_final.append((start, end, text))

    return final_final if final_final else [(words[0].start, words[-1].end, full_text)]



def _force_split_by_chars(
    text: str, start: float, end: float
) -> list[tuple[float, float, str]]:
    """
    纯字符数强制切分（最终兜底）。

    当 Whisper 的 word timestamps 全部被压缩到极短时间范围（1-2 秒）时，
    基于时间戳的切分和合并逻辑都会失效。这个函数不看时间戳，
    纯按字符数在日语标点/逗号/助词处切分，时间按字符比例均匀分配。
    """
    if len(text) <= _FORCE_SPLIT_CHARS:
        return [(start, end, text)]

    duration = end - start
    total_chars = len(text)

    # 优先在标点处切分
    # 日语标点优先级: 句号 > 逗号/顿号 > 读点 > 括号后 > 中间点
    _SPLIT_POINTS = set('。！？、，,. ）」』】》')
    _GOOD_SPLIT_POINTS = set('。！？')

    chunks = []
    chunk_start_char = 0

    i = 0
    while i < total_chars:
        remaining = total_chars - chunk_start_char
        if remaining <= _FORCE_SPLIT_CHARS:
            # 剩余部分不超限，直接收尾
            break

        # 在 [chunk_start + 30, chunk_start + _FORCE_SPLIT_CHARS] 范围内找最佳切分点
        search_start = chunk_start_char + 30  # 至少保证每段 30 字
        search_end = min(chunk_start_char + _FORCE_SPLIT_CHARS, total_chars)

        best_pos = -1
        # 先找句号类标点
        for p in range(search_end - 1, search_start - 1, -1):
            if text[p] in _GOOD_SPLIT_POINTS:
                best_pos = p + 1
                break
        # 再找逗号类
        if best_pos == -1:
            for p in range(search_end - 1, search_start - 1, -1):
                if text[p] in _SPLIT_POINTS:
                    best_pos = p + 1
                    break
        # 都找不到就硬切
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


def _rule_split_words(words_list) -> list[tuple[float, float, str]]:
    """
    纯规则切分一组词（用于对 wtpsplit 产出的超长句做二次切分）。
    规则与原始的 _split_segment_by_sentences 一致。
    """
    if not words_list:
        return []

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
        # 规则 2: 次级标点 + 超阈值
        elif word_text and word_text[-1] in _CLAUSE_ENDINGS:
            if current_duration >= 3.0 or current_chars >= 50:
                should_split = True
        # 规则 3: 连词前断
        elif (i + 1 < len(words_list)
              and current_duration >= 3.0
              and words_list[i + 1].word.strip().lower() in _BREAK_BEFORE_WORDS
              and current_chars >= 30):
            should_split = True
        # 规则 4: 硬限制
        elif current_duration >= _FORCE_SPLIT_DURATION or current_chars >= _FORCE_SPLIT_CHARS:
            should_split = True

        if should_split:
            _flush()
            if i + 1 < len(words_list):
                current_start = words_list[i + 1].start

    _flush()
    return results


def _split_with_rules(words, segment) -> list[tuple[float, float, str]]:
    """
    纯规则切分（wtpsplit 不可用时的 fallback）。
    逻辑与原始实现完全一致。
    """
    results = _rule_split_words(words)
    results = _merge_short_segments(results)
    # 兜底：强制切分超长条
    final = []
    for start, end, text in results:
        if len(text) > _FORCE_SPLIT_CHARS:
            final.extend(_force_split_by_chars(text, start, end))
        else:
            final.append((start, end, text))
    return final if final else [(segment.start, segment.end, segment.text.strip())]


def _merge_short_segments(
    segments: list[tuple[float, float, str]]
) -> list[tuple[float, float, str]]:
    """
    合并过短的字幕片段到相邻片段。

    规则：如果一个片段太短（字符数 < _MIN_CHARS 或时长 < _MIN_DURATION），
    将它合并到时间上更近的相邻片段（优先合并到后一个）。
    合并后时长不得超过 _MAX_DURATION **且** 字符数不得超过 _FORCE_SPLIT_CHARS，
    防止过度合并（尤其是 Whisper word timestamps 被压缩到极短范围的情况）。
    """
    if len(segments) <= 1:
        return segments

    merged = list(segments)
    changed = True

    def _can_merge(text_a: str, text_b: str, start: float, end: float) -> bool:
        """检查合并后是否在允许范围内（时长 + 字符数双重限制）"""
        return ((end - start) <= _MAX_DURATION
                and (len(text_a) + len(text_b)) <= _FORCE_SPLIT_CHARS)

    while changed:
        changed = False
        i = 0
        while i < len(merged):
            start, end, text = merged[i]
            char_count = len(text)
            duration = end - start

            is_short = (char_count < _MIN_CHARS
                        or duration < _MIN_DURATION)

            if is_short and len(merged) > 1:
                if i + 1 < len(merged):
                    # 尝试合并到后一个片段
                    next_start, next_end, next_text = merged[i + 1]
                    if _can_merge(text, next_text, start, next_end):
                        merged[i + 1] = (start, next_end, text + next_text)
                        merged.pop(i)
                        changed = True
                    else:
                        i += 1  # 合并后会超限，跳过
                elif i > 0:
                    # 最后一个短片段，尝试合并到前一个
                    prev_start, prev_end, prev_text = merged[i - 1]
                    if _can_merge(prev_text, text, prev_start, end):
                        merged[i - 1] = (prev_start, end, prev_text + text)
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
