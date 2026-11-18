# Ming-3.0-TTS-0.5B Non-Streaming Throughput (`rsi/ming3-tts-0.5b-nonstream`)

## Overview

`rsi/ming3-tts-0.5b-nonstream` is a reproducible GPU inference optimization benchmark in the [OpenRSI SGLang-Omni suite](../README.md). Agents optimize **[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/ca9e8789fcd84ed6f7272276cb99447eec751787)** (`ca9e8789fcd8`) serving throughput for **[Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B/tree/8d66b70a00f05ad183fa8a863b4ce609fc83cbc0)** on a single GPU under strict tail-latency, streaming-cadence, and speech-quality gates.

> [!NOTE]
> Archived reference trajectory (`Claude Code`) was recorded under a 12-hour single-agent budget on NVIDIA H100 NVL; default package budget is standardized to 24 hours (`86400s`).

## Official Single-Agent Reference Results

| Agent Harness | Best Verified Score (vs. $B_0$) | Best / Total Submissions | Wall Runtime | Trajectory Archive |
| :--- | ---: | ---: | ---: | :--- |
| Claude Code | **`4.298x`** | `agent-10` / `12` | `12.2 h` | [`sglang-omni/ming3-tts-0.5b-nonstream`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/ming3-tts-0.5b-nonstream) |

## Reference Baseline ($B_0$)

Reference baseline `B0` (measured on `NVIDIA H100 NVL`, 3-repetition mean per workload):

| Workload | Throughput (QPS) | Audio Throughput (s/s) | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER | TTFA Mean / P95 (s) | C50 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ming` | `8.767` | `34.653` | `1.813 / 2.641` | `0.4647 / 0.5883` | `0.0107` | — | — |

## Running with `rsi-harness`

This package runs out of the box with `rsi-harness`—no external wrapper scripts or manual pre-calibration steps are required on reference hardware:

```bash
# 1. Verify environment build and pristine baseline health (writes reward = 1.0)
rsi-harness run rsi-tasks/sglang-omni/ming3-tts-0.5b-nonstream --agent oracle

# 2. Launch an autonomous optimization run (24-hour default budget)
rsi-harness run rsi-tasks/sglang-omni/ming3-tts-0.5b-nonstream --agent codex --model <model-id>
```

### Hardware Isolation & Optional GPU Recalibration

- **Automatic NUMA & physical-core pinning**: When `RSI_TTS_CPU_LANES` is unset, the Judge automatically selects and binds **16 distinct non-SMT physical CPU cores** on the assigned GPU's NUMA node (preferring idle cores) and enforces `peak_foreign_cores <= 2` throughout evaluation.
- **Custom GPU SKUs (`RSI_BASELINE_JSON`)**: `tests/baseline.json` ships pre-calibrated for `NVIDIA H100 NVL`. To benchmark on a different GPU architecture (e.g., H100 PCIe/NVL, H200, A100, B200), measure a fresh 3-repetition `B0` on your target GPU using `/opt/sglang/bin/python3 -I -B /tests/calibrate.py` inside a container with `./tests` mounted at `/tests`, and pass the emitted `RSI_BASELINE_JSON='{...}'` environment variable to `rsi-harness run`.

## Package Layout

```text
ming3-tts-0.5b-nonstream/
├── task.toml        # Harbor v1.4 task & resource specification
├── instruction.md   # Self-contained Agent prompt and evaluation contract
├── README.md        # Human documentation, baseline table, and reproduction guide
├── environment/     # Pinned Dockerfile, lockfiles, and SGLang-Omni source archive
├── solution/        # Reference restoration script (used by --agent oracle)
└── tests/           # Clean-image Judge harness, gates, and reference baseline.json
```
