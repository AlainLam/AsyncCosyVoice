#!/usr/bin/env python3
"""CosyVoice TTS streaming inference performance test client.

Measures PCM first-packet latency across concurrency levels.

Usage:
    python example/test_client.py

Requirements:
    pip install openai httpx
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import httpx
from openai import AsyncOpenAI, Omit, omit

# ── Configuration ──────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
VOICE_ID = "test-voice"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
VOICE_WAV = ASSETS_DIR / "test.wav"
VOICE_TXT = ASSETS_DIR / "test.txt"
TMP_DIR = Path(__file__).parent.parent / "tmp"

# PCM format — confirmed from service config (cosyvoice3.yaml: sample_rate=24000,
# audio.py: tensor_to_pcm_s16le → int16 little-endian, mono output)
PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_BIT_DEPTH = 16  # s16le

# ── Test texts (1-indexed) ─────────────────────────────────────────────────────
# Each entry: (1-based index, text body, instruction or None)
TEXTS: list[tuple[int, str, str | Omit]] = [
    # Chinese — no instruction
    (1,  "晨光透过百叶窗，在地板上划出平行的金色条纹。",                             omit),
    (2,  "地铁门关闭的提示音响起，站台上的人群缓缓散去。",                           omit),
    # English — no instruction
    (3,  "The autumn leaves rustled under the footsteps of a passing jogger.",  omit),
    (4,  "A distant church bell echoed through the misty valley at dawn.",      omit),
    # Japanese
    (5,  "駅前の喫茶店で、温かいコーヒーカップが湯気を立てている。",               "请用日语说"),
    (6,  "雨上がりの夜道に、街灯の光が水たまりにゆらめいて見える。",               "请用日语说"),
    # Korean
    (7,  "아침 이슬이 맺힌 장미 꽃잎이 햇빛에 반짝이고 있다。",                   "请用韩语说"),
    (8,  "지하철 창밖으로 스쳐 지나가는 역명판이 흐릿하게 보인다。",               "请用韩语说"),
    # Cantonese
    (9,  "朝早七点，茶餐厅已经坐满人，到处都係叫点心同埋倾偈嘅声音。",             "请使用广东话说"),
    (10, "落紧微微雨，佢攥住把遮，企喺巴士站度等车。",                             "请使用广东话说"),
]

# ── Data containers ────────────────────────────────────────────────────────────
class RequestResult(NamedTuple):
    req_num: int               # 1-indexed within the round
    text_num: int              # 1-indexed text entry (1–10)
    first_packet_ms: float | None  # None on failure
    error: str | None


# ── Voice registration ────────────────────────────────────────────────────────
async def register_voice(http: httpx.AsyncClient) -> str:
    """Register the shared assets voice using a non-default voice ID."""
    ref_text = VOICE_TXT.read_text(encoding="utf-8").strip()
    with VOICE_WAV.open("rb") as wav_f:
        resp = await http.post(
            f"{BASE_URL}/v1/voices/register",
            data={"voice_id": VOICE_ID, "ref_text": ref_text},
            files={"ref_audio": (VOICE_WAV.name, wav_f, "audio/wav")},
            timeout=120.0,
        )
    if resp.status_code == 409:
        print(f"[INFO] Voice '{VOICE_ID}' already registered — reusing.")
        return VOICE_ID
    resp.raise_for_status()
    registered_id: str = resp.json()["voice_id"]
    print(f"[INFO] Voice registered: {registered_id}")
    return registered_id


# ── Single streaming request ──────────────────────────────────────────────────
async def run_request(
    oai: AsyncOpenAI,
    voice_id: str,
    req_num: int,
    text_num: int,
    text: str,
    instruction: str | Omit,
) -> tuple[RequestResult, bytes]:
    """Run one TTS streaming request.

    Returns (RequestResult, complete_pcm_bytes).
    First-packet latency is measured from the moment the HTTP request is
    dispatched until the first non-empty PCM chunk arrives.
    """
    pcm_chunks: list[bytes] = []
    first_packet_ms: float | None = None
    error: str | None = None

    try:
        t_start = time.perf_counter()
        async with oai.audio.speech.with_streaming_response.create(
            model="cosyvoice",
            voice=voice_id,
            input=text,
            response_format="pcm",
            instructions=instruction,
        ) as stream:
            async for chunk in stream.iter_bytes(chunk_size=4096):
                if not chunk:
                    continue
                if first_packet_ms is None:
                    first_packet_ms = (time.perf_counter() - t_start) * 1000
                pcm_chunks.append(chunk)
    except Exception as exc:
        error = repr(exc)
        print(f"    [ERROR] req{req_num} (text#{text_num}): {exc}")

    return (
        RequestResult(req_num=req_num, text_num=text_num,
                      first_packet_ms=first_packet_ms, error=error),
        b"".join(pcm_chunks),
    )


# ── One test round (N concurrent requests) ────────────────────────────────────
async def run_round(
    oai: AsyncOpenAI,
    voice_id: str,
    phase: str,      # "warmup" or "perf"
    concurrency: int,
    text_offset: int,  # rolling 0-based offset into TEXTS
) -> tuple[list[RequestResult], list[tuple[str, bytes]]]:
    """Send `concurrency` requests simultaneously.

    Returns:
        results    — one RequestResult per request
        file_pairs — list of (filename, pcm_bytes) for saving
    """
    coroutines = []
    text_indices: list[int] = []
    for i in range(concurrency):
        idx = (text_offset + i) % len(TEXTS)
        text_num, text, instruction = TEXTS[idx]
        text_indices.append(idx)
        coroutines.append(
            run_request(oai, voice_id, i + 1, text_num, text, instruction)
        )

    gathered = await asyncio.gather(*coroutines)

    results: list[RequestResult] = []
    file_pairs: list[tuple[str, bytes]] = []
    for i, (result, pcm_bytes) in enumerate(gathered):
        results.append(result)
        fname = (
            f"{phase}_c{concurrency}_req{result.req_num}_text{result.text_num}.pcm"
        )
        file_pairs.append((fname, pcm_bytes))

    return results, file_pairs


# ── Print round summary ───────────────────────────────────────────────────────
def print_round_summary(label: str, concurrency: int, results: list[RequestResult]) -> None:
    print(f"\n[{label}] Concurrency={concurrency}")
    latencies: list[float] = []
    for r in results:
        if r.first_packet_ms is not None:
            latencies.append(r.first_packet_ms)
            print(f"  req{r.req_num} (text#{r.text_num}): First-packet latency = {r.first_packet_ms:.0f}ms")
        else:
            print(f"  req{r.req_num} (text#{r.text_num}): FAILED — {r.error}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        print(
            f"  Round Statistics: min={min(latencies):.0f}ms  "
            f"max={max(latencies):.0f}ms  avg={avg:.0f}ms"
        )
    else:
        print("  Round Statistics: no successful requests")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # The OpenAI SDK appends /audio/speech to the base_url, so include /v1 here
    oai = AsyncOpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed")

    # Step 1: Register voice
    print("=" * 60)
    print("Step 1: Register voice")
    print("=" * 60)
    async with httpx.AsyncClient() as http:
        voice_id = await register_voice(http)

    all_pcm_files: list[tuple[str, bytes]] = []
    summary_rows: list[tuple[str, int, list[RequestResult]]] = []
    text_offset = 0  # rolling pointer so successive rounds use different texts

    # Step 2:  Warm-up Test ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Warm-up Test")
    print("=" * 60)
    for conc in [1, 2, 3, 4]:
        results, file_pairs = await run_round(
            oai, voice_id, "warmup", conc, text_offset
        )
        text_offset += conc
        all_pcm_files.extend(file_pairs)
        print_round_summary("Warm-up Test", conc, results)
        summary_rows.append(("Warm-up Test", conc, results))

    # Step 3: Formal Performance Test ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Formal Performance Test")
    print("=" * 60)
    for conc in [1, 2, 3, 4, 8]:
        results, file_pairs = await run_round(
            oai, voice_id, "perf", conc, text_offset
        )
        text_offset += conc
        all_pcm_files.extend(file_pairs)
        print_round_summary("Performance Test", conc, results)
        summary_rows.append(("Performance Test", conc, results))

    # Step 4: Save PCM + convert to WAV ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Save audio files")
    print("=" * 60)
    pcm_paths: list[Path] = []
    for fname, data in all_pcm_files:
        if data:
            p = TMP_DIR / fname
            p.write_bytes(data)
            pcm_paths.append(p)
            print(f"  Saved: {p.name}  ({len(data):,} bytes)")
        else:
            print(f"  Skipped: {fname}  (empty — request may have failed)")

    print(f"\nConverting {len(pcm_paths)} PCM file(s) → WAV …")
    for pcm_path in pcm_paths:
        wav_path = pcm_path.with_suffix(".wav")
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", str(PCM_SAMPLE_RATE),
            "-ac", str(PCM_CHANNELS),
            "-i", str(pcm_path),
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            pcm_path.unlink()
            print(f"  {pcm_path.name} → {wav_path.name}")
        else:
            print(
                f"  [WARN] ffmpeg failed for {pcm_path.name}:\n"
                f"         {proc.stderr.decode(errors='replace')[:300]}"
            )

    # Final summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    header = f"{'Phase':<22} {'Conc':>5}  {'Min':>7}  {'Max':>7}  {'Avg':>7}  {'OK/N':>5}"
    print(header)
    print("-" * 70)
    for label, conc, results in summary_rows:
        lats = [r.first_packet_ms for r in results if r.first_packet_ms is not None]
        n_total = len(results)
        n_ok = len(lats)
        if lats:
            row = (
                f"{label:<22} {conc:>5}  "
                f"{min(lats):>6.0f}ms  {max(lats):>6.0f}ms  "
                f"{sum(lats)/len(lats):>6.0f}ms  "
                f"{n_ok}/{n_total}"
            )
        else:
            row = f"{label:<22} {conc:>5}  {'—':>7}  {'—':>7}  {'—':>7}  {n_ok}/{n_total}"
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
