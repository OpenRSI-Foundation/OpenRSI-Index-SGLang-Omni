# Qwen3-TTS-1.7B Dual-Mode TTS Throughput

## Objective

Strictly improve **BOTH** `qwen17` (non-streaming) and `qwen17_stream` (streaming) English TTS throughput (`throughput_qps`) at concurrency 16 relative to the fixed reference baseline `B0`, maximizing the streaming-weighted geometric mean:

$$\text{Reward} = \left(\frac{\text{QPS}_{\text{stream}}(C)}{\text{QPS}_{\text{stream}}(B_0)}\right)^{2/3} \times \left(\frac{\text{QPS}_{\text{nonstream}}(C)}{\text{QPS}_{\text{nonstream}}(B_0)}\right)^{1/3}$$

Optimize default-on runtime execution—such as stage scheduling, batching, CUDA graph capture, CPU/GPU overlap, inter-stage IPC/relay transport, memory/KV reuse, or vocoder/codec execution—while preserving audio quality, tail latency, and request isolation.

## Baseline & Starting Environment

- **Runtime source**: SGLang-Omni (`1523543056bf334bc2f4d69db23aad0a002cedf2`) in `/workspace/sglang-omni` (clean restoration copy at `/opt/rsi-source`).
- **Model checkpoint**: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (`fd4b254389122332181a7c3db7f27e918eec64e3`), native `24000 Hz` `PCM_16` mono WAV output.
- **Authoritative ASR judge**: `Qwen/Qwen3-ASR-1.7B` (`7278e1e70fe206f11671096ffdd38061171dd6e5`), cached offline under `/opt/rsi-assets/huggingface`.
- **Model-specific defaults**: Native benchmark passes `--ref-format references`; request `max_new_tokens` cap is fixed at 2,048.
- **Benchmark CLI (non-streaming)**:
  ```bash
  /opt/sglang/bin/python3 -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 8000 --ref-format references --output-dir results/qwen17_en --lang en --max-concurrency 16
  ```
  *(Append `--stream` and use a separate `--output-dir` for streaming mode. Note that the benchmark harness passes `max_running_requests=64` and `cuda_graph_max_bs=64`; verify effective server settings before running experiments.)*

Reference baseline `B0` (measured on `NVIDIA H100 80GB HBM3`, 3-repetition mean per workload):

| Workload | Throughput (QPS) | Audio Throughput (s/s) | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER | TTFA Mean / P95 (s) | C50 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen17` | `18.790` | `78.038` | `0.847 / 1.174` | `0.2115 / 0.2946` | `0.0104` | — | — |
| `qwen17_stream` | `19.971` | `82.888` | `0.798 / 1.167` | `0.1942 / 0.2216` | `0.0105` | 0.098 / 0.134 | 99.68 |

## Workspace & Action Space

| Path | Role |
| :--- | :--- |
| `/workspace/sglang-omni` | Candidate runtime repository (deliverable) |
| `/workspace/research/` | Required research log directory (`hypothesis.md`, `experiments.json`, `reproduction.json`, `source.diff`) |
| `/opt/rsi-source` | Read-only pristine copy of original `/workspace/sglang-omni` |
| `/opt/sglang/bin/python3` | Python 3.12 runtime interpreter (offline mode) |
| `/opt/rsi-assets/huggingface` | Read-only Hugging Face model & SeedTTS dataset cache |

### Permitted vs. Protected Surfaces

- **Editable**: Runtime source files (`.py`, `.cu`, `.cuh`, `.cpp`, `.c`, `.h`, `.hpp`) under `/workspace/sglang-omni/sglang_omni/`, **except**:
  - Within `sglang_omni/models/`, **only** `sglang_omni/models/qwen3_tts/` may be modified.
  - `sglang_omni/models/qwen3_asr/` and `sglang_omni/sampling/` are **strictly protected** and must remain byte-identical.
- **Protected / Forbidden**:
  - Do not modify `benchmarks/`, `pyproject.toml`, tests, model checkpoints, tokenizers, sampling semantics, voice/reference conditioning, or output audio sample rate/format (`24000 Hz`, `PCM_16`, 1 channel).
  - No cross-request answer caching, SeedTTS sample ID/text lookup, audio truncation/resampling/time-stretching, or evaluator/environment probing. Note that repeated prompt/reference conditioning across distinct target utterances is normal voice-cloning usage, provided target synthesis remains faithful and request-isolated.
  - Do not leave generated `.pyc`, `__pycache__`, dotfiles, symlinks, or temporary benchmark outputs (`results/`) inside `/workspace/sglang-omni`, and do not add extra top-level entries under `/workspace` (only `/workspace/sglang-omni` and `/workspace/research` may exist at submission time).

## Required Research Deliverables

Every changed submission (`rsi-submit`) must include four non-empty files (each `<= 32 MiB`) inside `/workspace/research/`:

1. `hypothesis.md` — Falsifiable bottleneck hypothesis, profiling evidence, and causal mechanism.
2. `experiments.json` — Structured log of local benchmark measurements, ablations, and rejected/retained changes.
3. `reproduction.json` — Exact commands and configuration used to verify the candidate locally.
4. `source.diff` — Unified git diff of `/workspace/sglang-omni` against `/opt/rsi-source`.

*(Before calling `rsi-submit`, terminate all local server/GPU processes, remove any temporary benchmark output directories inside `/workspace/sglang-omni`, and flush all files.)*

## Fixed Verifier Protocol & Acceptance Gates

When you run `rsi-submit`, a fresh clean-image Judge container snapshots `/workspace` and executes the full evaluation against fixed `B0`:

| Parameter | Contract |
| :--- | :--- |
| Dataset | All 1,088 English rows of `zhaochenyang20/seed-tts-eval-arrow` (`27f4c1ad...`) in native order |
| Runs | 3 non-streaming runs (`qwen17`), then 3 streaming runs (`qwen17_stream`, `--stream`), serially on 1 GPU (6,528 scored requests + 96 warmups) |
| Seeds | Fixed repetition seeds `42, 43, 44` per mode; fresh server & empty compiler caches every repetition |
| Concurrency / Warmup | TTS concurrency `16`, TTS warmup `16` (excluded from throughput); authoritative ASR concurrency `32`, warmup `64` |
| Coverage | 1,088/1,088 valid WAVs and ASR transcripts per run; 0 failed or skipped requests |
| Hardware isolation | 1 GPU + 16 dedicated non-SMT physical CPU cores (`peak_foreign_cores <= 2`), 14,400s Judge timeout |

Let $C$ be your candidate's 3-repetition arithmetic mean and $B_0$ be the fixed reference baseline mean. **All 27 metric gates** must pass simultaneously:

- **Primary throughput (`throughput_qps`)**: Strict improvement **$C > B_0$** (in every evaluated mode).
- **Corpus quality (`wer_corpus`)**: **$C \le 0.02$** (absolute 2.0% micro-WER ceiling scored by unmodified `Qwen/Qwen3-ASR-1.7B`).
- **Latency & RTF guards**: `latency_{mean,median,p95,p99}_s` and `rtf_{mean,median,p95,p99}` must satisfy **$C \le 1.03 \times B_0$**; `audio_throughput_s_per_s` must satisfy **$C \ge 0.97 \times B_0$**.
- **Streaming playback continuity (`C50`) & chunking**: Every streaming request must use native `stream=true` PCM delivery and report valid `audio_ttfp_s` and per-chunk durations matching the decoded WAV. At least **95%** (`>= 1,034 / 1,088`) of streaming requests must emit multiple audio chunks (`audio_chunk_count > 1`) so single-chunk responses cannot bypass `C50` scoring. `C50` (percentage of multi-chunk requests with max playback underrun `<= 50 ms`) must satisfy `C >= 0.97 * B0`, and `audio_ttfp_{mean,median,p95,p99}_s` must satisfy `C <= 1.03 * B0`.

Only a candidate that passes every structural, coverage, audio-format, and performance/quality gate writes `/logs/verifier/reward.json`. Inspect detailed per-mode metrics and gate margins after each submission via `rsi-submit --list`.
