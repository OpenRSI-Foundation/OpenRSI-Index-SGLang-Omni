# Ming-3.0-TTS-Tiny Streaming Throughput (`rsi/ming3-tts-tiny-streaming`)

## Overview

`rsi/ming3-tts-tiny-streaming` is a reproducible GPU inference optimization benchmark in the [OpenRSI SGLang-Omni suite](../README.md). Agents optimize **[SGLang-Omni](https://github.com/sgl-project/sglang-omni/tree/0c663861d7d4a558e4f53258c4d48cfce577b47e)** (`0c663861d7d4`) serving throughput for **[Ming-3.0-tts-tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny/tree/7bd9446a8d47e462362261a0150258da6d458318)** on a single GPU under strict tail-latency, streaming-cadence, and speech-quality gates.

> [!NOTE]
> Archived reference trajectory (`codex`) was recorded under an 8-hour (`28800s`) sprint budget on NVIDIA H200; default package budget is standardized to 24 hours (`86400s`). Recalibrate `RSI_BASELINE_JSON` via `/tests/calibrate.py` when evaluating on non-H200 GPUs.

## Official Single-Agent Reference Results

| Agent Harness | Best Verified Score (vs. $B_0$) | Best / Total Submissions | Wall Runtime | Trajectory Archive |
| :--- | ---: | ---: | ---: | :--- |
| Codex | **`1.722x`** | `agent-10` / `10` | `8.0 h` | [`sglang-omni/ming3-tts-tiny-streaming`](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni/ming3-tts-tiny-streaming) |

## Reference Baseline ($B_0$)

Reference baseline `B0` (measured on `NVIDIA H200`, 3-repetition mean per workload):

| Workload | Throughput (QPS) | Audio Throughput (s/s) | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER | TTFA Mean / P95 (s) | C50 (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ming` | `5.313` | `19.083` | `2.993 / 4.389` | `0.8376 / 0.9568` | `0.0110` | 1.180 / 1.430 | 88.82 |

## Running with `rsi-harness`

This package runs out of the box with `rsi-harness`—no external wrapper scripts or manual pre-calibration steps are required on reference hardware:

```bash
# 1. Verify environment build and pristine baseline health (writes reward = 1.0)
rsi-harness run rsi-tasks/sglang-omni/ming3-tts-tiny-streaming --agent oracle

# 2. Launch an autonomous optimization run (24-hour default budget)
rsi-harness run rsi-tasks/sglang-omni/ming3-tts-tiny-streaming --agent codex --model <model-id>
```

### Hardware Isolation & Optional GPU Recalibration

- **Automatic NUMA & physical-core pinning**: When `RSI_TTS_CPU_LANES` is unset, the Judge automatically selects and binds **16 distinct non-SMT physical CPU cores** on the assigned GPU's NUMA node (preferring idle cores) and enforces `peak_foreign_cores <= 2` throughout evaluation.
- **Custom GPU SKUs (`RSI_BASELINE_JSON`)**: `tests/baseline.json` ships pre-calibrated for `NVIDIA H200`. To benchmark on a different GPU architecture (e.g., H100 PCIe/NVL, H200, A100, B200), measure a fresh 3-repetition `B0` on your target GPU using `/opt/sglang/bin/python3 -I -B /tests/calibrate.py` inside a container with `./tests` mounted at `/tests`, and pass the emitted `RSI_BASELINE_JSON='{...}'` environment variable to `rsi-harness run`.

## Package Layout

```text
ming3-tts-tiny-streaming/
├── task.toml        # Harbor v1.4 task & resource specification
├── instruction.md   # Self-contained Agent prompt and evaluation contract
├── README.md        # Human documentation, baseline table, and reproduction guide
├── environment/     # Pinned Dockerfile, lockfiles, and SGLang-Omni source archive
├── solution/        # Reference restoration script (used by --agent oracle)
└── tests/           # Clean-image Judge harness, gates, and reference baseline.json
```
