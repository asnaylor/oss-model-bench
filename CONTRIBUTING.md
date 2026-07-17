# Contributing

Keep this project thin. New orchestration layers, databases, dashboards, provider-specific adapters, and server launchers need evidence that the existing JSON/native-artifact workflow cannot solve the problem.

Before opening a change:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m oss_model_bench.cli perf --help
PYTHONPATH=src python -m oss_model_bench.cli agent --help
```

Changes to `panel-v1.json`, workload sizes, durations, concurrency, prompts, upstream tool versions, or dataset revisions change benchmark semantics. Introduce a new named panel or schema version rather than silently changing historical meaning. Record why the change improves signal within the two-hour target.

Never commit endpoint credentials, raw private prompts, cloned task repositories, or generated benchmark results. Preserve native upstream artifacts and clearly label subset scores as non-official.
