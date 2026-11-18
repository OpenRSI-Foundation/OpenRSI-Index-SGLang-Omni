# Qwen3-ASR-1.7B Dual-Mode Streaming & Non-Streaming Throughput

## Objective

Improve fixed-weight **Qwen3-ASR-1.7B** serving throughput on a single GPU across **both** post-upload SSE streaming (`stream/en`, `stream/zh`) and non-streaming (`nonstream/en`, `nonstream/zh`) OpenAI-compatible `/v1/audio/transcriptions` workloads relative to the fixed reference baseline `B0`.

For each mode $m \in \{\text{stream}, \text{nonstream}\}$, define the bilingual geometric-mean throughput ratio:

$$R_m(C; B_0) = \sqrt{\frac{Q_{m,\text{en}}(C)}{Q_{m,\text{en}}(B_0)} \cdot \frac{Q_{m,\text{zh}}(C)}{Q_{m,\text{zh}}(B_0)}}$$

Maximize the streaming-weighted geometric-mean reward:

$$\text{Reward} = R_{\text{stream}}(C; B_0)^{2/3} \cdot R_{\text{nonstream}}(C; B_0)^{1/3}$$

## Baseline & Starting Environment

- **Runtime source**: SGLang-Omni (`eb7e97ed9438ef4e4352656aec6170aeeb90966f`) in `/workspace/sglang-omni` (pristine restoration logic in `/opt/asr-reference/restore.py`).
- **Model checkpoint**: `Qwen/Qwen3-ASR-1.7B` (`7278e1e70fe206f11671096ffdd38061171dd6e5`) at `/opt/models/qwen3-asr-1.7b`.
- **Fixed serving semantics**: BF16 (`asr.factory.dtype=bfloat16`), `enable_torch_compile=True`, `disable_radix_cache=True`, `enable_pre_lm_encoder=True`, `pre_lm_cache_size_bytes=1`, `mm_embedding_cache_size_bytes=0`, `stream_emit_interval_s=0.05`, `max_new_tokens=128`, greedy decoding, and cold-input encoder cache hooks disabled.

Reference baseline `B0` (measured on `NVIDIA H100 80GB HBM3`, 3-repetition mean per workload at HTTP concurrency 32):

| Workload | QPS | RTFx | Latency Mean / P95 (s) | RTF Mean / P95 | Corpus WER/CER | TTFT Mean / P95 (s) | Inter-Chunk Mean / P95 (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nonstream/en` | `168.92` | `792.61` | `0.1879 / 0.2436` | `0.0413 / 0.0572` | `0.0121` | — | — |
| `nonstream/zh` | `179.78` | `841.68` | `0.1767 / 0.2232` | `0.0382 / 0.0490` | `0.0061` | — | — |
| `stream/en` | `160.60` | `753.58` | `0.1967 / 0.2573` | `0.0432 / 0.0594` | `0.0121` | 0.1018 / 0.1365 | 0.0481 / 0.0793 |
| `stream/zh` | `172.06` | `805.53` | `0.1842 / 0.2386` | `0.0398 / 0.0516` | `0.0061` | 0.1029 / 0.1305 | 0.0470 / 0.0777 |

## Workspace & Action Space

| Path | Role |
| :--- | :--- |
| `/workspace/sglang-omni` | Candidate SGLang-Omni source tree (deliverable) |
| `/workspace/candidate.yaml` | Optional flat YAML mapping of execution knobs, initially `{}` (deliverable) |
| `/workspace/research/` | Research notes, profiling logs, and synthetic audio diagnostics |
| `/opt/models/qwen3-asr-1.7b` | Read-only model weights and tokenizer |
| `/opt/asr-venv/bin/python` | Python 3.12 virtual environment (`sglang_omni.cli serve`) |

### What You May Modify

1. **Runtime source directories** under `/workspace/sglang-omni/sglang_omni/`:
   `models/qwen3_asr`, `scheduling`, `comm`, `http`, `serve`, `model_runner`, `pipeline`, `relay`, `sampling`, `utils`, and `kernels` (regular `.py` files `<= 8 MiB`).
   - Within these directories, AST structural gates lock the fixed audio preprocessing, language resolution, prompt/token-budget functions, greedy sampling assignments, and cold-input cache-disable methods (`return None` / `return False`), and verify those functions remain actively wired.
2. **Execution configuration (`/workspace/candidate.yaml`, `<= 64 KiB`)**:
   - Permitted factory keys: `asr.factory.pre_lm_max_batch_size`, `asr.factory.pre_lm_max_batch_wait_ms`, `asr.factory.request_build_max_workers`, `asr.factory.request_build_max_pending`, `asr.factory.prefill_coalesce_requests`, `asr.factory.prefill_coalesce_wait_ms`, `asr.factory.prefill_coalesce_when_idle`, `asr.factory.prefill_coalesce_requires_pending_builds`, `asr.factory.prefill_coalesce_after_builds_during_decode`.
   - Permitted engine keys: `asr.engine.max_running_requests`, `asr.engine.torch_compile_max_bs`, `asr.engine.cuda_graph_max_bs`, `asr.engine.disable_cuda_graph`, `asr.engine.mem_fraction_static`, `asr.engine.kv_cache_bytes`.
   - Do not set both `mem_fraction_static` (`0 < x < 1`) and `kv_cache_bytes` (`1 <= x < 2^63`). `prefill_coalesce_requests` must be `0` or `>= 2`; `prefill_coalesce_wait_ms > 0`.

### What Stays Fixed / Forbidden

- Do not modify files outside the permitted directories above, model weights, Python environment (`/opt/asr-venv`), or `/opt/asr-reference`.
- No cross-request caching of embeddings, features, transcripts, or audio-dependent KV state; no evaluation/test-path probing; and no full-transcript buffering disguised as SSE streaming.
- Keep all synthetic test scripts and diagnostic artifacts strictly inside `/workspace/research/`. Before calling `rsi-submit`, stop all local server/GPU processes and flush all files.

## Fixed Verifier Protocol & Acceptance Gates

Each `rsi-submit` launches a clean-image Judge container that evaluates **12 fresh-service passes** (`nonstream/en`, `nonstream/zh`, `stream/en`, `stream/zh` $\times$ 3 repetitions = **18,648 warmup + 18,648 scored HTTP requests** at closed-loop concurrency `32`, 300s per-request timeout, 10,800s total Judge timeout):

- Every pass launches a fresh `sglang_omni.cli serve` instance on `127.0.0.1:18080` with empty compiler caches (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`), completes 1 full dataset warmup pass (`1,088` EN or `2,020` ZH clips), synchronizes CUDA, and measures 1 full scored pass.
- **SSE streaming contract (`stream/*`)**: Responses must use `Content-Type: text/event-stream`, emit ordered non-empty `transcript.text.delta` events whose concatenation equals `transcript.text.done` followed by `[DONE]`, produce non-null inter-chunk intervals, and satisfy **`single_delta_fraction <= 0.05`** (preventing single-chunk whole-transcript buffering).
- **Mode throughput gains**: Both **$R_{\text{stream}}(C; B_0) > 1$** and **$R_{\text{nonstream}}(C; B_0) > 1$** are strictly required, and every individual workload's `qps` and `rtfx` must satisfy **$C \ge 0.97 \times B_0$**.
- **Latency & streaming cadence guards**: In every workload, `latency_mean_s`, `latency_p95_s`, `rtf_mean`, `rtf_p95` (plus `ttft_mean_s`, `ttft_p95_s`, `inter_chunk_mean_s`, `inter_chunk_p95_s` on `stream/*`) must satisfy **$C \le 1.03 \times B_0$**.
- **Accuracy gate**: Corpus WER/CER mean `<= 0.02` (absolute 2.0% ceiling) independently in each of the 4 workloads (`nonstream/en`, `nonstream/zh`, `stream/en`, `stream/zh`).

Only a complete evaluation with zero failed gates writes `/logs/verifier/reward.json`. Use `rsi-submit --list` to inspect repeat-level and aggregate metrics after each submission.
