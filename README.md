# oss-model-bench

`oss-model-bench` is a small, endpoint-agnostic regression suite for an OSS model served behind an OpenAI-compatible API. It answers two different questions without mixing their scores:

1. **Can the deployment serve agent-shaped traffic quickly?** NVIDIA AIPerf measures output tokens/s, request throughput, TTFT, inter-token latency, tail latency, and errors.
2. **Can the model actually do agent work?** BFCL measures tool calling, while OpenCode attempts pinned repository issue-resolution tasks that are graded by the official SWE-bench harness.

It does not launch vLLM, reserve GPUs, know about a particular platform, or invent a composite “agent score.” Point it at an already-running endpoint.

## Default suite

| Command | Workload | Default wall time |
| --- | --- | ---: |
| `omb check` | Chat, streaming, forced tool call | < 2 min |
| `omb perf` | 1k/256 and 4k/512 at concurrency 1, 4, 16; then synthetic agentic sessions | 15–25 min |
| `omb agent` | 50 BFCL cases and 6 OpenCode/SWE-bench tasks, 4 workers | 45–90 min |

The performance suite includes agentic context because context growth, prefix reuse, and KV-cache pressure are serving behavior. It remains a distinct section inside the AIPerf result, not a second benchmark framework.

The agentic trace grows to 80% of `OMB_CONTEXT_LIMIT`, capped at 200,000 tokens. A 200k test is useful when the deployed model genuinely supports it, but running every throughput point at 200k would be slow and would obscure ordinary agent latency.

## Install

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra all
```

Install [OpenCode](https://opencode.ai/docs/) separately and ensure `opencode` is on `PATH`. The Python extras install [NVIDIA AIPerf](https://docs.nvidia.com/aiperf/), [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), Hugging Face Datasets, and [SWE-bench](https://www.swebench.com/SWE-bench/guides/evaluation/).

## Configure a target

```bash
export OMB_BASE_URL=https://model.example/v1
export OMB_MODEL=served-model-name
export OMB_TOKENIZER=org/tokenizer-or-local-path
export OMB_CONTEXT_LIMIT=131072
export OMB_RESULTS_DIR=$PWD/results
# Optional when the endpoint requires bearer authentication:
export OMB_API_KEY=replace-me
```

`OMB_TOKENIZER` defaults to `OMB_MODEL`. `OMB_API_KEY` is optional. `OPENAI_BASE_URL` and `OPENAI_API_KEY` are accepted as fallbacks. Secrets are redacted from summaries and command records, and result files are mode `0600`. When authentication is enabled, AIPerf receives the key as a CLI argument, so it can be briefly visible to other users with permission to inspect the benchmark process table; use a short-lived, least-privilege credential on shared systems.

Run the compatibility gate first:

```bash
uv run omb check
```

The check fails unless chat completions, streaming, and a forced function call all succeed. It allows 256 completion tokens for chat and 512 for the forced tool call so that reasoning models have room to emit their final structured answer; a reasoning-only response ending with `finish_reason: length` fails the gate. `GET /models` is reported but is not a hard requirement. Tool-call parser selection (for example, vLLM's `qwen3_coder`) is a server setting and is not configured by OMB.

## Performance

```bash
uv run omb perf
```

The baseline runs six independent AIPerf profiles:

- chat: 1,024 input / 256 output tokens;
- code: 4,096 input / 512 output tokens;
- concurrency: 1, 4, and 16 for each shape;
- 15-second warmup and 60-second measurement per profile.

It then uses AIPerf's Agentic Code generator to create four deterministic multi-turn sessions and replays them as a Mooncake trace. This exercises shared system/tool prefixes, repository context, growing history, and KV-cache reuse at concurrency 1 and 4 for five minutes.

Useful shorter commands are:

```bash
uv run omb perf --phase baseline
uv run omb perf --phase agentic --agentic-duration 120
uv run omb perf --dry-run
```

Treat output tokens/s together with TTFT, p90/p99 latency, inter-token latency, and error rate. Maximizing tokens/s alone can make an interactive coding agent worse.

## Agent and SWE capability

```bash
uv run omb agent --workers 4
```

The bundled `panel-v1` is deliberately small and versioned. BFCL uses exact case IDs. SWE tasks use indexes from a pinned revision of the SWE-bench Lite `dev` split. OpenCode receives a temporary inline provider configuration and edits one clean checkout per task. The produced patches use the official SWE-bench prediction format.

BFCL has an upstream portability constraint: its `--model` value must name a handler registered by BFCL. If the endpoint's served name is different, provide the compatible BFCL handler alias and configure the server/gateway to accept that alias:

```bash
export OMB_BFCL_MODEL=the-supported-bfcl-handler-name
uv run omb agent
```

This is a BFCL limitation, not an endpoint coupling in this repo. Check BFCL's `SUPPORTED_MODELS.md`; a truly new prompt/tool format requires an upstream model handler.

SWE-bench grading requires a working Docker-compatible daemon. If the generation node lacks one, preserve the patches and skip grading:

```bash
uv run omb agent --no-grade
```

Then copy `results/agent-*/predictions.jsonl` to a Docker host and run the official harness there. `--no-bfcl` and `--no-swe` isolate either capability layer. `--task-timeout 900` sets a per-issue OpenCode limit.

## Results and comparisons

Every invocation writes a timestamped directory with `summary.json`, logs, native tool artifacts, and—where applicable—patches and predictions. Native AIPerf, BFCL, and SWE-bench files remain the source of truth.

```text
results/
├── check.json
├── perf-.../
│   ├── summary.json
│   ├── logs/
│   └── native/aiperf/
└── agent-.../
    ├── summary.json
    ├── native/bfcl/
    ├── patches/
    ├── predictions.jsonl
    └── tasks/
```

Compare two runs without a database or dashboard:

```bash
uv run omb compare results/perf-old/summary.json results/perf-new/summary.json
```

The comparison reports numeric deltas for matching AIPerf fields. For capability results, consult native BFCL accuracy and SWE-bench resolved counts. The bundled subset is a regression panel, not an official leaderboard submission, and cross-run comparisons are only meaningful with the same panel, tool versions, hardware, server settings, and warm-cache policy.

## Slurm / NERSC

The benchmark client is CPU-only. On a development cluster with negligible queue time, keep model serving and benchmarking as separate jobs: start the existing model deployment, wait for `omb check`, then submit the benchmark client. A heterogeneous Slurm allocation adds coupling without improving the measurement because this repo does not own the server lifecycle.

Prepare the environment and caches on a login node, then:

```bash
sbatch -A YOUR_ACCOUNT scripts/nersc-cpu-agent.slurm
```

The supplied script runs capability generation with `--no-grade`. Podman-HPC is not automatically a drop-in replacement for the Docker API expected by the official SWE-bench harness; grade elsewhere unless you have explicitly provided a compatible Docker service.

## Scope and limitations

- This is a fast regression suite, not an official BFCL or SWE-bench leaderboard run.
- Six SWE tasks are useful for catching catastrophic regressions, not for estimating a stable population-level resolve rate. Run larger official subsets for release claims.
- Repository setup is intentionally thin: OpenCode works in a checkout and may install or run tests if the task permits, while correctness is decided later in the official SWE-bench image.
- Long-context synthesis tests serving mechanics, not whether the model understands every token.
- Endpoint access, model deployment, GPU telemetry, power, and cost accounting are outside the portable core.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m oss_model_bench.cli --help
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before changing the pinned panel or benchmark defaults.
