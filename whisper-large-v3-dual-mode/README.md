# Whisper-large-v3 Dual-Mode Streaming & Non-Streaming Throughput (`rsi/whisper-large-v3-dual-mode`)

## Overview

`rsi/whisper-large-v3-dual-mode` is a reproducible GPU inference optimization benchmark in the [OpenRSI SGLang-Omni suite](../README.md). Agents optimize **[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/eb7e97ed9438ef4e4352656aec6170aeeb90966f)** (`eb7e97ed9438`) serving throughput for **[Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3/tree/06f233fe06e710322aca913c1bc4249a0d71fce1)** on a single GPU under strict tail-latency, streaming-cadence, and speech-quality gates.

## Official Single-Agent Reference Results

| Agent Harness | Best Verified Score (vs. $B_0$) | Best / Total Submissions | Wall Runtime | Trajectory Archive |
| :--- | ---: | ---: | ---: | :--- |
| Claude Code | **`1.979x`** | `agent-12` / `12` | `24.0 h` | [`sglang-omni/whisper-large-v3-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/whisper-large-v3-dual-mode) |
| Codex | **`1.126x`** | `agent-8` / `16` | `25.7 h` | [`sglang-omni/whisper-large-v3-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/whisper-large-v3-dual-mode) |

## Reference Baseline ($B_0$)

Reference baseline `B0` (measured on `NVIDIA H100 80GB HBM3`, 3-repetition mean per workload at HTTP concurrency 32):

| Workload | QPS | RTFx | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER/CER | TTFT Mean / P95 (s) | Inter-Chunk Mean / P95 (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nonstream/en` | `53.21` | `249.69` | `0.5977 / 0.7484` | `0.1311 / 0.1789` | `0.0137` | — | — |
| `nonstream/zh` | `48.62` | `227.65` | `0.6570 / 0.8786` | `0.1418 / 0.1897` | `0.0663` | — | — |
| `stream/en` | `52.43` | `246.02` | `0.6066 / 0.7404` | `0.1330 / 0.1820` | `0.0137` | 0.2109 / 0.2990 | 0.0982 / 0.2530 |
| `stream/zh` | `48.57` | `227.39` | `0.6577 / 0.8922` | `0.1420 / 0.1932` | `0.0663` | 0.2078 / 0.3398 | 0.0923 / 0.2345 |

## Running with `rsi-harness`

This package runs out of the box with `rsi-harness`—no external wrapper scripts or manual pre-calibration steps are required on reference hardware:

```bash
# 1. Verify environment build and pristine baseline health (writes reward = 1.0)
rsi-harness run rsi-tasks/sglang-omni/whisper-large-v3-dual-mode --agent oracle

# 2. Launch an autonomous optimization run (24-hour default budget)
rsi-harness run rsi-tasks/sglang-omni/whisper-large-v3-dual-mode --agent codex --model <model-id>
```

### Hardware Isolation & Optional GPU Recalibration

- **Automatic NUMA & physical-core pinning**: When `RSI_ASR_CPU_LANES` is unset, the Judge automatically selects and binds **16 distinct non-SMT physical CPU cores** on the assigned GPU's NUMA node (preferring idle cores) and enforces `peak_foreign_cores <= 2` throughout evaluation.
- **Custom GPU SKUs (`RSI_BASELINE_JSON`)**: `tests/baseline.json` ships pre-calibrated for `NVIDIA H100 80GB HBM3`. To benchmark on a different GPU architecture (e.g., H100 PCIe/NVL, H200, A100, B200), measure a fresh 3-repetition `B0` on your target GPU using `/opt/asr-venv/bin/python -I -B /tests/calibrate.py` inside a container with `./tests` mounted at `/tests`, and pass the emitted `RSI_BASELINE_JSON='{...}'` environment variable to `rsi-harness run`.

## Package Layout

```text
whisper-large-v3-dual-mode/
├── task.toml        # Harbor v1.4 task & resource specification
├── instruction.md   # Self-contained Agent prompt and evaluation contract
├── README.md        # Human documentation, baseline table, and reproduction guide
├── environment/     # Pinned Dockerfile, lockfiles, and SGLang-Omni source archive
├── solution/        # Reference restoration script (used by --agent oracle)
└── tests/           # Clean-image Judge harness, gates, and reference baseline.json
```
