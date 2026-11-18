# Ming-3.0-TTS-0.5B Non-Streaming Throughput

## Objective

Strictly improve `ming` (non-streaming) English TTS throughput (`throughput_qps`) at concurrency 16 relative to the fixed reference baseline `B0`, maximizing the throughput speedup ratio:

$$\text{Reward} = \frac{\text{QPS}(C)}{\text{QPS}(B_0)}$$

Optimize default-on runtime execution—such as stage scheduling, batching, CUDA graph capture, CPU/GPU overlap, inter-stage IPC/relay transport, memory/KV reuse, or vocoder/codec execution—while preserving audio quality, tail latency, and request isolation.

## Baseline & Starting Environment

- **Runtime source**: SGLang-Omni (`ca9e8789fcd84ed6f7272276cb99447eec751787`) in `/workspace/sglang-omni` (clean restoration copy at `/opt/rsi-source`).
- **Model checkpoint**: `inclusionAI/Ming-3.0-tts-0.5B` (`8d66b70a00f05ad183fa8a863b4ce609fc83cbc0`), native `44100 Hz` `PCM_16` mono WAV output.
- **Authoritative ASR judge**: `Qwen/Qwen3-ASR-1.7B` (`7278e1e70fe206f11671096ffdd38061171dd6e5`), cached offline under `/opt/rsi-assets/huggingface`.
- **Model-specific defaults**: Native 44.1 kHz latent diffusion + VAE audio synthesis; 1,000 max latent steps, temperature 0, minimum 5 decode steps, request `max_new_tokens` cap 2,048.
- **Benchmark CLI (non-streaming)**:
  ```bash
  /opt/sglang/bin/python3 -m benchmarks.eval.benchmark_tts_seedtts --meta zhaochenyang20/seed-tts-eval-arrow --model inclusionAI/Ming-3.0-tts-0.5B --port 8000 --output-dir results/ming3_tts_en --lang en --max-concurrency 16
  ```
  *(Append `--stream` and use a separate `--output-dir` for streaming mode. Note that the benchmark harness passes `max_running_requests=64` and `cuda_graph_max_bs=64`; verify effective server settings before running experiments.)*

Reference baseline `B0` (measured on `NVIDIA H100 NVL`, 3-repetition mean per workload):

| Workload | Throughput (QPS) | Audio Throughput (s/s) | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER | TTFA Mean / P95 (s) | C50 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ming` | `8.767` | `34.653` | `1.813 / 2.641` | `0.4647 / 0.5883` | `0.0107` | — | — |

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
  - Within `sglang_omni/models/`, **only** `sglang_omni/models/ming3_tts/` may be modified.
  - `sglang_omni/models/qwen3_asr/` and `sglang_omni/sampling/` are **strictly protected** and must remain byte-identical.
- **Protected / Forbidden**:
  - Do not modify `benchmarks/`, `pyproject.toml`, tests, model checkpoints, tokenizers, sampling semantics, voice/reference conditioning, or output audio sample rate/format (`44100 Hz`, `PCM_16`, 1 channel).
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
| Runs | 3 serial non-streaming runs (`ming`) on 1 GPU (3,264 scored requests + 48 warmups) |
| Seeds | Fixed repetition seeds `42, 43, 44` per mode; fresh server & empty compiler caches every repetition |
| Concurrency / Warmup | TTS concurrency `16`, TTS warmup `16` (excluded from throughput); authoritative ASR concurrency `32`, warmup `64` |
| Coverage | 1,088/1,088 valid WAVs and ASR transcripts per run; 0 failed or skipped requests |
| Hardware isolation | 1 GPU + 16 dedicated non-SMT physical CPU cores (`peak_foreign_cores <= 2`), 14,400s Judge timeout |

Let $C$ be your candidate's 3-repetition arithmetic mean and $B_0$ be the fixed reference baseline mean. **All 11 metric gates** must pass simultaneously:

- **Primary throughput (`throughput_qps`)**: Strict improvement **$C > B_0$** (in every evaluated mode).
- **Corpus quality (`wer_corpus`)**: **$C \le 0.02$** (absolute 2.0% micro-WER ceiling scored by unmodified `Qwen/Qwen3-ASR-1.7B`).
- **Latency & RTF guards**: `latency_{mean,median,p95,p99}_s` and `rtf_{mean,median,p95,p99}` must satisfy **$C \le 1.03 \times B_0$**; `audio_throughput_s_per_s` must satisfy **$C \ge 0.97 \times B_0$**.

Only a candidate that passes every structural, coverage, audio-format, and performance/quality gate writes `/logs/verifier/reward.json`. Inspect detailed per-mode metrics and gate margins after each submission via `rsi-submit --list`.
