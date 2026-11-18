# Qwen3-TTS-1.7B Dual-Mode TTS Throughput (`rsi/qwen3-tts-1.7b-dual-mode`)

## Overview

`rsi/qwen3-tts-1.7b-dual-mode` is a reproducible GPU inference optimization benchmark in the [OpenRSI SGLang-Omni suite](../README.md). Agents optimize **[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/1523543056bf334bc2f4d69db23aad0a002cedf2)** (`1523543056bf`) serving throughput for **[Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base/tree/fd4b254389122332181a7c3db7f27e918eec64e3)** on a single GPU under strict tail-latency, streaming-cadence, and speech-quality gates.

## Official Single-Agent Reference Results

| Agent Harness | Best Verified Score (vs. $B_0$) | Best / Total Submissions | Wall Runtime | Trajectory Archive |
| :--- | ---: | ---: | ---: | :--- |
| Claude Code | **`1.283x`** | `agent-3` / `3` | `24.1 h` | [`sglang-omni/qwen3-tts-1.7b-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/qwen3-tts-1.7b-dual-mode) |
| Codex | **`1.096x`** | `agent-10` / `13` | `24.1 h` | [`sglang-omni/qwen3-tts-1.7b-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/qwen3-tts-1.7b-dual-mode) |

## Reference Baseline ($B_0$)

Reference baseline `B0` (measured on `NVIDIA H100 80GB HBM3`, 3-repetition mean per workload):

| Workload | Throughput (QPS) | Audio Throughput (s/s) | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER | TTFA Mean / P95 (s) | C50 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen17` | `18.790` | `78.038` | `0.847 / 1.174` | `0.2115 / 0.2946` | `0.0104` | — | — |
| `qwen17_stream` | `19.971` | `82.888` | `0.798 / 1.167` | `0.1942 / 0.2216` | `0.0105` | 0.098 / 0.134 | 99.68 |

## Running with `rsi-harness`

This package runs out of the box with `rsi-harness`—no external wrapper scripts or manual pre-calibration steps are required on reference hardware:

```bash
# 1. Verify environment build and pristine baseline health (writes reward = 1.0)
rsi-harness run rsi-tasks/sglang-omni/qwen3-tts-1.7b-dual-mode --agent oracle

# 2. Launch an autonomous optimization run (24-hour default budget)
rsi-harness run rsi-tasks/sglang-omni/qwen3-tts-1.7b-dual-mode --agent codex --model <model-id>
```

### Hardware Isolation & Optional GPU Recalibration

- **Automatic NUMA & physical-core pinning**: When `RSI_TTS_CPU_LANES` is unset, the Judge automatically selects and binds **16 distinct non-SMT physical CPU cores** on the assigned GPU's NUMA node (preferring idle cores) and enforces `peak_foreign_cores <= 2` throughout evaluation.
- **Custom GPU SKUs (`RSI_BASELINE_JSON`)**: `tests/baseline.json` ships pre-calibrated for `NVIDIA H100 80GB HBM3`. To benchmark on a different GPU architecture (e.g., H100 PCIe/NVL, H200, A100, B200), measure a fresh 3-repetition `B0` on your target GPU using `/opt/sglang/bin/python3 -I -B /tests/calibrate.py` inside a container with `./tests` mounted at `/tests`, and pass the emitted `RSI_BASELINE_JSON='{...}'` environment variable to `rsi-harness run`.

## Package Layout

```text
qwen3-tts-1.7b-dual-mode/
├── task.toml        # Harbor v1.4 task & resource specification
├── instruction.md   # Self-contained Agent prompt and evaluation contract
├── README.md        # Human documentation, baseline table, and reproduction guide
├── environment/     # Pinned Dockerfile, lockfiles, and SGLang-Omni source archive
├── solution/        # Reference restoration script (used by --agent oracle)
└── tests/           # Clean-image Judge harness, gates, and reference baseline.json
```
