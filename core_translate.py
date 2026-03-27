"""
core_translate.py - LWF 翻译核心引擎

基于 OpenAI Chat Completions 兼容 API，支持：
- DeepSeek / OpenAI GPT / Claude (兼容端点) / Ollama / LM Studio
- 分批翻译 + 编号锚定（防丢行）
- 自动重试 + 逐句降级
- 进度回调

用户只需配置: api_base + api_key + model_name
"""

import re
import json
import time
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("LWF")

# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TranslateResult:
    """单句翻译结果"""
    index: int          # 对应原始 segment 序号
    source: str         # 原文
    target: str         # 译文（翻译失败时为空字符串）
    success: bool = True


@dataclass
class TranslateConfig:
    """翻译 API 配置"""
    api_base: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    target_lang: str = "简体中文"
    source_lang: str = ""       # 空 = 自动（从转录语言推断）
    batch_size: int = 20        # 每批翻译句数
    max_retries: int = 3        # 单批重试次数
    timeout: int = 60           # 单次请求超时秒数
    temperature: float = 0.3    # 翻译温度（低 = 稳定）
    bilingual: bool = False     # 是否输出双语


# ═══════════════════════════════════════════════════════════════════════════
# 预设 API 服务商
# ═══════════════════════════════════════════════════════════════════════════

API_PROVIDERS = {
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "OpenAI": {
        "api_base": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano"],
    },
    "Ollama (本地)": {
        "api_base": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b",
        "models": ["qwen2.5:7b", "qwen2.5:14b", "llama3:8b"],
    },
    "自定义": {
        "api_base": "",
        "default_model": "",
        "models": [],
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# 翻译 Prompt
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """你是一个专业字幕翻译器。你的任务是翻译字幕文本。

严格规则：
1. 输入格式为编号行，如 [1] 原文、[2] 原文...
2. 输出必须严格保持相同编号格式：[1] 译文、[2] 译文...
3. 行数必须与输入完全一致，不多不少
4. 翻译要自然流畅，符合目标语言口语习惯
5. 保留专有名词（人名、地名、品牌名）
6. 不要添加任何解释、注释或额外内容
7. 每个编号占一行，不要合并或拆分"""


def _build_user_prompt(texts: list[str], source_lang: str, target_lang: str) -> str:
    """构建用户 prompt"""
    lang_hint = f"从{source_lang}" if source_lang else ""
    header = f"请将以下字幕{lang_hint}翻译成{target_lang}：\n\n"
    numbered = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    return header + numbered


# ═══════════════════════════════════════════════════════════════════════════
# API 调用（纯 stdlib，不依赖 openai/httpx）
# ═══════════════════════════════════════════════════════════════════════════

def _call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    system_msg: str,
    user_msg: str,
    temperature: float = 0.3,
    timeout: int = 60,
) -> str:
    """
    调用 OpenAI 兼容的 Chat Completions API。
    使用 urllib 实现，零外部依赖。
    返回 assistant 回复文本。
    """
    # 规范化 endpoint
    base = api_base.rstrip("/")
    if not base.endswith("/chat/completions"):
        base = base.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(base, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise TranslateAPIError(
            f"API 返回 HTTP {e.code}: {error_body[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise TranslateAPIError(
            f"无法连接到 API 端点 {api_base}: {e.reason}"
        ) from e
    except Exception as e:
        raise TranslateAPIError(f"API 请求失败: {e}") from e


class TranslateAPIError(Exception):
    """翻译 API 错误"""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# 结果解析（编号锚定）
# ═══════════════════════════════════════════════════════════════════════════

_NUM_PATTERN = re.compile(r"^\[(\d+)\]\s*(.*)$")


def _parse_numbered_response(response: str, expected_count: int) -> dict[int, str]:
    """
    解析带编号的翻译结果。
    返回 {编号: 译文} 字典。

    示例输入:
        [1] 大家早上好
        [2] 今天的新闻

    返回: {1: "大家早上好", 2: "今天的新闻"}
    """
    result = {}
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _NUM_PATTERN.match(line)
        if m:
            idx = int(m.group(1))
            text = m.group(2).strip()
            result[idx] = text

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 翻译引擎
# ═══════════════════════════════════════════════════════════════════════════

class TranslationEngine:
    """
    字幕翻译引擎。
    支持任何 OpenAI Chat Completions 兼容 API。
    """

    def __init__(self, config: TranslateConfig):
        self.config = config

    def translate_batch(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
        batch_done_callback: Callable[[int, int, list[TranslateResult]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[TranslateResult]:
        """
        批量翻译文本列表。

        Args:
            texts: 要翻译的文本列表（对应 SRT 每句的文本）
            progress_callback: 进度回调 (completed, total, current_text)
            batch_done_callback: 每批完成后回调 (batch_start, batch_end, batch_results)
                                 batch_results 是这一批对应位置的 TranslateResult 列表
            cancel_check: 取消检查函数，返回 True 时中止

        Returns:
            TranslateResult 列表，与输入 texts 一一对应
        """
        cfg = self.config
        total = len(texts)
        results: list[TranslateResult] = [
            TranslateResult(index=i, source=t, target="", success=False)
            for i, t in enumerate(texts)
        ]

        # 分批
        batches = []
        for start in range(0, total, cfg.batch_size):
            end = min(start + cfg.batch_size, total)
            batches.append((start, end))

        completed = 0

        for batch_start, batch_end in batches:
            # 取消检查
            if cancel_check and cancel_check():
                logger.info("[TRANSLATE] 用户取消翻译")
                break

            batch_texts = texts[batch_start:batch_end]
            batch_size = len(batch_texts)

            logger.info(
                f"[TRANSLATE] 翻译批次 {batch_start+1}-{batch_end}/{total}"
            )

            # 尝试整批翻译
            translated = self._translate_batch_with_retry(
                batch_texts, batch_start
            )

            if translated is not None:
                # 整批成功
                for i, text in enumerate(batch_texts):
                    global_idx = batch_start + i
                    local_num = i + 1  # 编号从 1 开始
                    if local_num in translated:
                        results[global_idx].target = translated[local_num]
                        results[global_idx].success = True
                    else:
                        # 这句没拿到翻译，逐句补救
                        single = self._translate_single_with_retry(text)
                        if single:
                            results[global_idx].target = single
                            results[global_idx].success = True
                        else:
                            logger.warning(
                                f"[TRANSLATE] 第 {global_idx+1} 句翻译失败，保留原文"
                            )
                            results[global_idx].target = text  # 保留原文
                            results[global_idx].success = False
            else:
                # 整批失败，逐句降级
                logger.warning(
                    f"[TRANSLATE] 批次 {batch_start+1}-{batch_end} 整批失败，逐句降级"
                )
                for i, text in enumerate(batch_texts):
                    if cancel_check and cancel_check():
                        break

                    global_idx = batch_start + i
                    single = self._translate_single_with_retry(text)
                    if single:
                        results[global_idx].target = single
                        results[global_idx].success = True
                    else:
                        results[global_idx].target = text
                        results[global_idx].success = False

                    completed += 1
                    if progress_callback:
                        progress_callback(
                            completed, total,
                            f"[降级] {global_idx+1}/{total}"
                        )
                # 降级批次完成后也回调
                if batch_done_callback:
                    batch_results = results[batch_start:batch_end]
                    batch_done_callback(batch_start, batch_end, batch_results)
                continue  # 跳过下面的 completed 更新

            completed += batch_size
            if batch_done_callback:
                batch_results = results[batch_start:batch_end]
                batch_done_callback(batch_start, batch_end, batch_results)
            if progress_callback:
                preview = ""
                # 取最后一句有翻译的作为预览
                for i in range(batch_end - 1, batch_start - 1, -1):
                    if results[i].success:
                        preview = results[i].target[:40]
                        break
                progress_callback(completed, total, preview)

        return results

    def translate_single(self, text: str) -> str | None:
        """翻译单句，返回译文或 None"""
        return self._translate_single_with_retry(text)

    def test_connection(self) -> tuple[bool, str]:
        """
        测试 API 连接。
        返回 (success, message)。
        """
        try:
            reply = _call_chat_api(
                api_base=self.config.api_base,
                api_key=self.config.api_key,
                model=self.config.model,
                system_msg="Reply with exactly: OK",
                user_msg="Test connection",
                temperature=0,
                timeout=15,
            )
            return True, f"连接成功！模型回复: {reply[:50]}"
        except TranslateAPIError as e:
            return False, str(e)
        except Exception as e:
            return False, f"未知错误: {e}"

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _translate_batch_with_retry(
        self, texts: list[str], global_offset: int
    ) -> dict[int, str] | None:
        """
        翻译一个 batch，带重试。
        返回 {局部编号: 译文} 字典，或 None（全部重试失败）。
        """
        cfg = self.config
        user_prompt = _build_user_prompt(
            texts, cfg.source_lang, cfg.target_lang
        )

        for attempt in range(1, cfg.max_retries + 1):
            try:
                response = _call_chat_api(
                    api_base=cfg.api_base,
                    api_key=cfg.api_key,
                    model=cfg.model,
                    system_msg=_SYSTEM_PROMPT,
                    user_msg=user_prompt,
                    temperature=cfg.temperature,
                    timeout=cfg.timeout,
                )

                parsed = _parse_numbered_response(response, len(texts))

                # 校验：至少拿到 80% 的行才算成功
                if len(parsed) >= len(texts) * 0.8:
                    if len(parsed) < len(texts):
                        missing = set(range(1, len(texts)+1)) - set(parsed.keys())
                        logger.warning(
                            f"[TRANSLATE] 批次缺少编号: {missing}，将逐句补救"
                        )
                    return parsed
                else:
                    logger.warning(
                        f"[TRANSLATE] 批次解析不足: "
                        f"期望 {len(texts)} 行，仅解析 {len(parsed)} 行 "
                        f"(尝试 {attempt}/{cfg.max_retries})"
                    )

            except TranslateAPIError as e:
                logger.warning(
                    f"[TRANSLATE] API 错误 (尝试 {attempt}/{cfg.max_retries}): {e}"
                )

            except Exception as e:
                logger.warning(
                    f"[TRANSLATE] 未知错误 (尝试 {attempt}/{cfg.max_retries}): {e}"
                )

            # 指数退避
            if attempt < cfg.max_retries:
                wait = 2 ** attempt
                logger.info(f"[TRANSLATE] 等待 {wait} 秒后重试...")
                time.sleep(wait)

        return None

    def _translate_single_with_retry(self, text: str) -> str | None:
        """翻译单句，带重试"""
        cfg = self.config
        user_prompt = _build_user_prompt(
            [text], cfg.source_lang, cfg.target_lang
        )

        for attempt in range(1, cfg.max_retries + 1):
            try:
                response = _call_chat_api(
                    api_base=cfg.api_base,
                    api_key=cfg.api_key,
                    model=cfg.model,
                    system_msg=_SYSTEM_PROMPT,
                    user_msg=user_prompt,
                    temperature=cfg.temperature,
                    timeout=30,
                )

                parsed = _parse_numbered_response(response, 1)
                if 1 in parsed and parsed[1]:
                    return parsed[1]

                # 如果没有编号格式，直接取第一行非空内容
                for line in response.splitlines():
                    line = line.strip()
                    if line and not line.startswith("["):
                        return line

            except Exception as e:
                logger.warning(
                    f"[TRANSLATE] 单句重试 {attempt}/{cfg.max_retries}: {e}"
                )
                if attempt < cfg.max_retries:
                    time.sleep(1)

        return None


# ═══════════════════════════════════════════════════════════════════════════
# SRT 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def parse_srt(srt_text: str) -> list[dict]:
    """
    解析 SRT 文本为结构化列表。

    返回: [{"index": 1, "start": "00:00:01,000", "end": "00:00:03,000", "text": "..."},  ...]
    """
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    segments = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not time_match:
            continue

        text = "\n".join(lines[2:]).strip()

        segments.append({
            "index": index,
            "start": time_match.group(1),
            "end": time_match.group(2),
            "text": text,
        })

    return segments


def build_srt(
    segments: list[dict],
    translations: list[TranslateResult] | None = None,
    bilingual: bool = False,
) -> str:
    """
    重建 SRT 文本。

    Args:
        segments: parse_srt() 的输出
        translations: 翻译结果列表（与 segments 一一对应）
        bilingual: 是否双语（原文 + 译文各一行）

    Returns:
        SRT 格式文本
    """
    lines = []

    for i, seg in enumerate(segments):
        lines.append(str(seg["index"]))
        lines.append(f"{seg['start']} --> {seg['end']}")

        if translations and i < len(translations):
            tr = translations[i]
            if bilingual:
                lines.append(seg["text"])
                lines.append(tr.target if tr.target else seg["text"])
            else:
                lines.append(tr.target if tr.target else seg["text"])
        else:
            lines.append(seg["text"])

        lines.append("")  # 空行分隔

    return "\n".join(lines)


def build_srt_from_segments(
    segment_infos,  # list[SegmentInfo] from core_transcribe
    translations: list[TranslateResult],
    bilingual: bool = False,
) -> str:
    """
    从 core_transcribe 的 SegmentInfo 列表 + 翻译结果构建 SRT。
    用于转录后直接翻译的场景。
    """
    lines = []

    for i, seg in enumerate(segment_infos):
        lines.append(str(seg.index))
        lines.append(seg.timestamp_srt)  # "HH:MM:SS,mmm --> HH:MM:SS,mmm"

        if i < len(translations):
            tr = translations[i]
            if bilingual:
                lines.append(seg.text.strip())
                lines.append(tr.target if tr.target else seg.text.strip())
            else:
                lines.append(tr.target if tr.target else seg.text.strip())
        else:
            lines.append(seg.text.strip())

        lines.append("")

    return "\n".join(lines)
