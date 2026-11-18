# OpenRSI × SGLang-Omni Benchmark Suite

**Task repository:** [OpenRSI-Foundation/OpenRSI-Index-SGLang-Omni](https://github.com/OpenRSI-Foundation/OpenRSI-Index-SGLang-Omni) · **Parent index:** [OpenRSI-Index](https://github.com/OpenRSI-Foundation/OpenRSI-Index) (`rsi-tasks/sglang-omni/`) · **Trajectories:** [OpenRSI-Index-Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni) · **Blog:** [OpenRSI × SGLang-Omni](https://index.openrsi.foundation/blog/2026/openrsi-sglang-omni/)

This repository hosts **12 native Harbor 1.4 GPU inference optimization benchmarks** for [RSI-Harness](https://github.com/OpenRSI-Foundation/OpenRSI-Index/tree/main/RSI-Harness), built on [SGLang-Omni](https://github.com/sgl-project/sglang-omni) across **9 Text-to-Speech (TTS)** tasks and **3 Automatic Speech Recognition / Diarization (ASR)** tasks spanning 8 open-weight model checkpoints.

Every task runs out of the box with `rsi-harness run`:
- **Self-contained Docker build**: Pinned CUDA/SGLang dependencies, deterministic source manifests, and SHA-256-verified Hugging Face weights and SeedTTS evaluation splits.
- **Clean-image multi-repetition Judge**: Fresh server processes and empty compiler caches per repetition, automatic 16-physical-core NUMA affinity binding, host CPU contention monitoring (`peak_foreign_cores <= 2`), strict throughput improvement over fixed reference `B0`, `3%` tail-latency/TTFA/TTFT/continuity guards, and authoritative WER/CER verification.
- **Out-of-the-box `--agent oracle` verification**: Restores pristine baseline code, verifies full end-to-end execution health, and emits `reward = 1.0`.

## Benchmark Catalog

| Task Package | Modality | Model Checkpoint | Workload Mode | Reference $B_0$ GPU | Codex | Claude Code | Trajectories |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| [`cosyvoice3-dual-mode`](./cosyvoice3-dual-mode/) | TTS | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.172x` | `1.297x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/cosyvoice3-dual-mode) |
| [`moss-tts-v1.5-dual-mode`](./moss-tts-v1.5-dual-mode/) | TTS | `OpenMOSS-Team/MOSS-TTS-v1.5` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.339x` | `1.935x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/moss-tts-v1.5-dual-mode) |
| [`fish-s2-pro-dual-mode`](./fish-s2-pro-dual-mode/) | TTS | `fishaudio/s2-pro` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.803x` | `6.288x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/fish-s2-pro-dual-mode) |
| [`qwen3-tts-0.6b-dual-mode`](./qwen3-tts-0.6b-dual-mode/) | TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.141x` | `1.252x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/qwen3-tts-0.6b-dual-mode) |
| [`qwen3-tts-1.7b-dual-mode`](./qwen3-tts-1.7b-dual-mode/) | TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.096x` | `1.283x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/qwen3-tts-1.7b-dual-mode) |
| [`ming3-tts-0.5b-nonstream`](./ming3-tts-0.5b-nonstream/) | TTS | `inclusionAI/Ming-3.0-tts-0.5B` | Non-Streaming | `NVIDIA H100 NVL` | — | `4.298x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/ming3-tts-0.5b-nonstream) |
| [`ming3-tts-0.5b-streaming`](./ming3-tts-0.5b-streaming/) | TTS | `inclusionAI/Ming-3.0-tts-0.5B` | Streaming | `NVIDIA H200` | `2.038x` | — | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/ming3-tts-0.5b-streaming) |
| [`ming3-tts-tiny-nonstream`](./ming3-tts-tiny-nonstream/) | TTS | `inclusionAI/Ming-3.0-tts-tiny` | Non-Streaming | `NVIDIA H100 NVL` | `1.899x` | `3.634x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/ming3-tts-tiny-nonstream) |
| [`ming3-tts-tiny-streaming`](./ming3-tts-tiny-streaming/) | TTS | `inclusionAI/Ming-3.0-tts-tiny` | Streaming | `NVIDIA H200` | `1.722x` | — | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/ming3-tts-tiny-streaming) |
| [`qwen3-asr-1.7b-dual-mode`](./qwen3-asr-1.7b-dual-mode/) | ASR | `Qwen/Qwen3-ASR-1.7B` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.117x` | `1.317x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/qwen3-asr-1.7b-dual-mode) |
| [`whisper-large-v3-dual-mode`](./whisper-large-v3-dual-mode/) | ASR | `openai/whisper-large-v3` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.126x` | `1.979x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/whisper-large-v3-dual-mode) |
| [`moss-transcribe-diarize-dual-mode`](./moss-transcribe-diarize-dual-mode/) | ASR | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | Dual-Mode (Stream + Non-Stream) | `NVIDIA H100 80GB HBM3` | `1.134x` | `1.490x` | [Logs](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/moss-transcribe-diarize-dual-mode) |

## Quick Start

You can run any benchmark either from a recursive `OpenRSI-Index` checkout or directly from a standalone clone of this repository:

```bash
# Option A: via OpenRSI-Index submodule
git clone --recurse-submodules https://github.com/OpenRSI-Foundation/OpenRSI-Index.git
cd OpenRSI-Index
rsi-harness run rsi-tasks/sglang-omni/cosyvoice3-dual-mode --agent oracle

# Option B: standalone task repository clone
git clone https://github.com/OpenRSI-Foundation/OpenRSI-Index-SGLang-Omni.git
rsi-harness run OpenRSI-Index-SGLang-Omni/qwen3-asr-1.7b-dual-mode --agent oracle
```

See each task's `README.md` for hardware recalibration options (`RSI_BASELINE_JSON` via `/tests/calibrate.py`) when benchmarking on a GPU SKU different from the recorded reference `B0`.
