# MOSS-Transcribe-Diarize Dual-Mode Streaming & Non-Streaming Throughput (`rsi/moss-transcribe-diarize-dual-mode`)

## Overview

`rsi/moss-transcribe-diarize-dual-mode` is a reproducible GPU inference optimization benchmark in the [OpenRSI SGLang-Omni suite](../README.md). Agents optimize **[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/8220d220703cf8d8f5c5d9a2cde00b5ed13717b8)** (`8220d220703c`) serving throughput for **[MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/tree/704aa4a9c304e8520be88901e0d1960158ef5b15)** on a single GPU under strict tail-latency, streaming-cadence, and speech-quality gates.

## Official Single-Agent Reference Results

| Agent Harness | Best Verified Score (vs. $B_0$) | Best / Total Submissions | Wall Runtime | Trajectory Archive |
| :--- | ---: | ---: | ---: | :--- |
| Claude Code | **`1.490x`** | `agent-6` / `11` | `24.0 h` | [`sglang-omni/moss-transcribe-diarize-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/moss-transcribe-diarize-dual-mode) |
| Codex | **`1.134x`** | `agent-19` / `22` | `24.7 h` | [`sglang-omni/moss-transcribe-diarize-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/moss-transcribe-diarize-dual-mode) |

## Reference Baseline ($B_0$)

Reference baseline `B0` (measured on `NVIDIA H100 80GB HBM3`, 3-repetition mean per workload at HTTP concurrency 32):

| Workload | QPS | RTFx | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER/CER | TTFT Mean / P95 (s) | Inter-Chunk Mean / P95 (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nonstream/en` | `106.81` | `501.16` | `0.2976 / 0.3593` | `0.0655 / 0.0880` | `0.0183` | — | — |
| `nonstream/zh` | `109.60` | `513.12` | `0.2904 / 0.3552` | `0.0628 / 0.0780` | `0.0104` | — | — |
| `stream/en` | `102.15` | `479.33` | `0.3095 / 0.3728` | `0.0681 / 0.0920` | `0.0183` | 0.0856 / 0.1258 | 0.0547 / 0.0791 |
| `stream/zh` | `105.50` | `493.92` | `0.3012 / 0.3675` | `0.0652 / 0.0807` | `0.0104` | 0.0863 / 0.1269 | 0.0546 / 0.0796 |

## Running with `rsi-harness`

This package runs out of the box with `rsi-harness`—no external wrapper scripts or manual pre-calibration steps are required on reference hardware:

```bash
# 1. Verify environment build and pristine baseline health (writes reward = 1.0)
rsi-harness run rsi-tasks/sglang-omni/moss-transcribe-diarize-dual-mode --agent oracle

# 2. Launch an autonomous optimization run (24-hour default budget)
rsi-harness run rsi-tasks/sglang-omni/moss-transcribe-diarize-dual-mode --agent codex --model <model-id>
```

### Hardware Isolation & Optional GPU Recalibration

- **Automatic NUMA & physical-core pinning**: When `RSI_ASR_CPU_LANES` is unset, the Judge automatically selects and binds **16 distinct non-SMT physical CPU cores** on the assigned GPU's NUMA node (preferring idle cores) and enforces `peak_foreign_cores <= 2` throughout evaluation.
- **Custom GPU SKUs (`RSI_BASELINE_JSON`)**: `tests/baseline.json` ships pre-calibrated for `NVIDIA H100 80GB HBM3`. To benchmark on a different GPU architecture (e.g., H100 PCIe/NVL, H200, A100, B200), measure a fresh 3-repetition `B0` on your target GPU using `/opt/asr-venv/bin/python -I -B /tests/calibrate.py` inside a container with `./tests` mounted at `/tests`, and pass the emitted `RSI_BASELINE_JSON='{...}'` environment variable to `rsi-harness run`.

## Package Layout

```text
moss-transcribe-diarize-dual-mode/
├── task.toml        # Harbor v1.4 task & resource specification
├── instruction.md   # Self-contained Agent prompt and evaluation contract
├── README.md        # Human documentation, baseline table, and reproduction guide
├── environment/     # Pinned Dockerfile, lockfiles, and SGLang-Omni source archive
├── solution/        # Reference restoration script (used by --agent oracle)
└── tests/           # Clean-image Judge harness, gates, and reference baseline.json
```
