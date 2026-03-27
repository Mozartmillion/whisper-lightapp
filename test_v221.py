"""v2.2.1 CJK 助词断句修复 — 快速测试脚本"""
import sys, os, time

# 确保 utils_env 先初始化
sys.path.insert(0, os.path.dirname(__file__))
import utils_env  # noqa: F401 — 必须先于其他 import

from core_transcribe import extract_audio, transcribe_stream, segments_to_srt

VIDEO = (
    r"C:\Users\mozartliu\Downloads\NHK おはよう日本　3月21日(土) 米 中東に海兵隊派遣 "
    r"先読めぬ情勢・どうなる石油安定供給 歴史に鑑み解説・90歳のSNS「生の証残したい」 他"
    r"\1-NHK おはよう日本　3月21日(土) 米 中東に海兵隊派遣 先読めぬ情勢・どうなる石油安定供給 "
    r"歴史に鑑み解説・90歳のSNS「生の証残したい」 他-480P 标清-AVC.mp4"
)

OUTPUT_SRT = os.path.join(os.path.dirname(VIDEO), "v221-test.srt")

def main():
    env = utils_env.init_environment()
    print(f"[TEST] Output: {OUTPUT_SRT}")

    # Phase 1: 提取音频
    print("[TEST] Phase 1: extracting audio...")
    t0 = time.time()
    wav_path = extract_audio(VIDEO, env)
    print(f"[TEST] audio extracted: {time.time()-t0:.1f}s")

    # Phase 2: 转录
    print("[TEST] Phase 2: transcribing...")
    t1 = time.time()
    segments = []
    for seg in transcribe_stream(
        wav_path=wav_path,
        env=env,
        model_path="large-v3",
        device="cuda",
        language=None,         # auto detect
        beam_size=5,
        vad_filter=True,
        vad_min_silence_ms=500,
    ):
        segments.append(seg)
        # 打印进度
        if seg.index <= 5 or seg.index % 20 == 0:
            print(f"  #{seg.index} [{seg.start_fmt}->{seg.end_fmt}] {seg.text[:60]}")

    elapsed = time.time() - t1
    print(f"[TEST] done: {len(segments)} segments, {elapsed:.1f}s")

    # Phase 3: 保存 SRT
    srt_text = segments_to_srt(segments)
    with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
        f.write(srt_text)
    print(f"[TEST] SRT saved: {OUTPUT_SRT}")

    # 简单统计
    lengths = [len(s.text) for s in segments]
    print(f"\n[TEST] Stats:")
    print(f"  total: {len(segments)}")
    print(f"  avg chars: {sum(lengths)/len(lengths):.1f}")
    print(f"  max: {max(lengths)}")
    print(f"  min: {min(lengths)}")
    print(f"  over 45: {sum(1 for l in lengths if l > 45)}")

if __name__ == "__main__":
    main()
