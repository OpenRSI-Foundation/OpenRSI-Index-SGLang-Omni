# Qwen3-ASR-1.7B Dual-Mode Streaming & Non-Streaming Throughput (`rsi/qwen3-asr-1.7b-dual-mode`)

## Overview

`rsi/qwen3-asr-1.7b-dual-mode` is a reproducible GPU inference optimization benchmark in the [OpenRSI SGLang-Omni suite](../README.md). Agents optimize **[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/eb7e97ed9438ef4e4352656aec6170aeeb90966f)** (`eb7e97ed9438`) serving throughput for **[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5)** on a single GPU under strict tail-latency, streaming-cadence, and speech-quality gates.

## Official Single-Agent Reference Results

| Agent Harness | Best Verified Score (vs. $B_0$) | Best / Total Submissions | Wall Runtime | Trajectory Archive |
| :--- | ---: | ---: | ---: | :--- |
| Claude Code | **`1.317x`** | `agent-13` / `13` | `24.4 h` | [`sglang-omni/qwen3-asr-1.7b-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/qwen3-asr-1.7b-dual-mode) |
| Codex | **`1.117x`** | `agent-47` / `50` | `24.5 h` | [`sglang-omni/qwen3-asr-1.7b-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/qwen3-asr-1.7b-dual-mode) |

## Reference Baseline ($B_0$)

Reference baseline `B0` (measured on `NVIDIA H100 80GB HBM3`, 3-repetition mean per workload at HTTP concurrency 32):

| Workload | QPS | RTFx | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER/CER | TTFT Mean / P95 (s) | Inter-Chunk Mean / P95 (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nonstream/en` | `168.92` | `792.61` | `0.1879 / 0.2436` | `0.0413 / 0.0572` | `0.0121` | — | — |
| `nonstream/zh` | `179.78` | `841.68` | `0.1767 / 0.2232` | `0.0382 / 0.0490` | `0.0061` | — | — |
| `stream/en` | `160.60` | `753.58` | `0.1967 / 0.2573` | `0.0432 / 0.0594` | `0.0121` | 0.1018 / 0.1365 | 0.0481 / 0.0793 |
| `stream/zh` | `172.06` | `805.53` | `0.1842 / 0.2386` | `0.0398 / 0.0516` | `0.0061` | 0.1029 / 0.1305 | 0.0470 / 0.0777 |

## Running with `rsi-harness`

This package runs out of the box with `rsi-harness`—no external wrapper scripts or manual pre-calibration steps are required on reference hardware:

```bash
# 1. Verify environment build and pristine baseline health (writes reward = 1.0)
rsi-harness run rsi-tasks/sglang-omni/qwen3-asr-1.7b-dual-mode --agent oracle

# 2. Launch an autonomous optimization run (24-hour default budget)
rsi-harness run rsi-tasks/sglang-omni/qwen3-asr-1.7b-dual-mode --agent codex --model <model-id>
```

### Hardware Isolation & Optional GPU Recalibration

- **Automatic NUMA & physical-core pinning**: When `RSI_ASR_CPU_LANES` is unset, the Judge automatically selects and binds **16 distinct non-SMT physical CPU cores** on the assigned GPU's NUMA node (preferring idle cores) and enforces `peak_foreign_cores <= 2` throughout evaluation.
- **Custom GPU SKUs (`RSI_BASELINE_JSON`)**: `tests/baseline.json` ships pre-calibrated for `NVIDIA H100 80GB HBM3`. To benchmark on a different GPU architecture (e.g., H100 PCIe/NVL, H200, A100, B200), measure a fresh 3-repetition `B0` on your target GPU using `/opt/asr-venv/bin/python -I -B /tests/calibrate.py` inside a container with `./tests` mounted at `/tests`, and pass the emitted `RSI_BASELINE_JSON='{...}'` environment variable to `rsi-harness run`.

## Package Layout

```text
qwen3-asr-1.7b-dual-mode/
├── task.toml        # Harbor v1.4 task & resource specification
├── instruction.md   # Self-contained Agent prompt and evaluation contract
├── README.md        # Human documentation, baseline table, and reproduction guide
├── environment/     # Pinned Dockerfile, lockfiles, and SGLang-Omni source archive
├── solution/        # Reference restoration script (used by --agent oracle)
└── tests/           # Clean-image Judge harness, gates, and reference baseline.json
```
