# MOSS-TTS-v1.5 Dual-Mode TTS Throughput (`rsi/moss-tts-v1.5-dual-mode`)

## Overview

`rsi/moss-tts-v1.5-dual-mode` is a reproducible GPU inference optimization benchmark in the [OpenRSI SGLang-Omni suite](../README.md). Agents optimize **[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/cb75ab409669a6ab672fa42421b41b11af2fe263)** (`cb75ab409669`) serving throughput for **[MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5/tree/cdd3b911b1585e3f2dbc7775ef10f9926f58850a)** on a single GPU under strict tail-latency, streaming-cadence, and speech-quality gates.

## Official Single-Agent Reference Results

| Agent Harness | Best Verified Score (vs. $B_0$) | Best / Total Submissions | Wall Runtime | Trajectory Archive |
| :--- | ---: | ---: | ---: | :--- |
| Claude Code | **`1.935x`** | `agent-4` / `4` | `24.0 h` | [`sglang-omni/moss-tts-v1.5-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/moss-tts-v1.5-dual-mode) |
| Codex | **`1.339x`** | `agent-8` / `10` | `24.1 h` | [`sglang-omni/moss-tts-v1.5-dual-mode`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/moss-tts-v1.5-dual-mode) |

## Reference Baseline ($B_0$)

Reference baseline `B0` (measured on `NVIDIA H100 80GB HBM3`, 3-repetition mean per workload):

| Workload | Throughput (QPS) | Audio Throughput (s/s) | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER | TTFA Mean / P95 (s) | C50 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `moss` | `7.862` | `35.227` | `2.026 / 2.551` | `0.4663 / 0.6023` | `0.0176` | — | — |
| `moss_stream` | `6.560` | `29.392` | `2.429 / 3.054` | `0.5589 / 0.7156` | `0.0180` | 1.065 / 1.290 | 90.98 |

## Running with `rsi-harness`

This package runs out of the box with `rsi-harness`—no external wrapper scripts or manual pre-calibration steps are required on reference hardware:

```bash
# 1. Verify environment build and pristine baseline health (writes reward = 1.0)
rsi-harness run rsi-tasks/sglang-omni/moss-tts-v1.5-dual-mode --agent oracle

# 2. Launch an autonomous optimization run (24-hour default budget)
rsi-harness run rsi-tasks/sglang-omni/moss-tts-v1.5-dual-mode --agent codex --model <model-id>
```

### Hardware Isolation & Optional GPU Recalibration

- **Automatic NUMA & physical-core pinning**: When `RSI_TTS_CPU_LANES` is unset, the Judge automatically selects and binds **16 distinct non-SMT physical CPU cores** on the assigned GPU's NUMA node (preferring idle cores) and enforces `peak_foreign_cores <= 2` throughout evaluation.
- **Custom GPU SKUs (`RSI_BASELINE_JSON`)**: `tests/baseline.json` ships pre-calibrated for `NVIDIA H100 80GB HBM3`. To benchmark on a different GPU architecture (e.g., H100 PCIe/NVL, H200, A100, B200), measure a fresh 3-repetition `B0` on your target GPU using `/opt/sglang/bin/python3 -I -B /tests/calibrate.py` inside a container with `./tests` mounted at `/tests`, and pass the emitted `RSI_BASELINE_JSON='{...}'` environment variable to `rsi-harness run`.

## Package Layout

```text
moss-tts-v1.5-dual-mode/
├── task.toml        # Harbor v1.4 task & resource specification
├── instruction.md   # Self-contained Agent prompt and evaluation contract
├── README.md        # Human documentation, baseline table, and reproduction guide
├── environment/     # Pinned Dockerfile, lockfiles, and SGLang-Omni source archive
├── solution/        # Reference restoration script (used by --agent oracle)
└── tests/           # Clean-image Judge harness, gates, and reference baseline.json
```
