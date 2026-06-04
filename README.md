
# Setup

## RL Agent Environemnt
```bash
conda env create -f setup/environment.yml
conda activate atlasrl
```

## Smoke-test the RL loop
This validates the loop without starting the TRExFitter container:

```bash
python -m agent.smoke_test_env
```

Run the fuller tester:
```bash
python agent/test.py --clean
```

The smoke test uses `scripts/mock_trex.py`, which writes `results.json` and prints
`SIGNIFICANCE=...`. The environment then parses that significance and computes:

```text
reward = significance - target_significance
```

## verl LLM RL
Create the prompt dataset:
```bash
python -m atlasrl_verl.prepare_data --repeats 32
```

Smoke-test the verl reward function without starting verl:
```bash
python -m atlasrl_verl.test_reward
```

Install verl in the active training environment, then launch the starter GRPO run:
```bash
pip install verl
bash verl_configs/atlasrl_grpo_smoke.sh
```

The reward function expects the model to output one JSON edit action:
```json
{"op": "replace", "line": 17, "text": "  DebugLevel: 1", "n": 1}
```

By default the reward uses `scripts/mock_trex.py`. Set `ATLASRL_VERL_RUNNER=real`
or put `"runner": "real"` in `extra_info` when you are ready to score with
containerized TRExFitter.

## TRExFitter Environment
Pull image
```bash
podman-hpc pull gitlab-registry.cern.ch/atlas/statanalysis:0-4
```

Run image
```bash
podman-hpc run -it --rm \
  --group-add keep-groups \
  -v "$PWD":/workdir:rw \
  -v /global/cfs/projectdirs/atlas/haichen/opendata:/global/cfs/projectdirs/atlas/haichen/opendata:ro \
  -w /workdir \
  gitlab-registry.cern.ch/atlas/statanalysis:0-4
```

Check the wrapper:
```bash
python scripts/trex.py configs/hyy.config --check
```

Run histogram creation in parallel by splitting the `n` step by region:
```bash
python scripts/trex.py configs/hyy.config \
  --actions n \
  --parallel-regions \
  --region-workers 6 \
  --regions-per-job 1
```

Then run the remaining TRExFitter stages after all region jobs finish:
```bash
python scripts/trex.py configs/hyy.config --actions w f s
```

Keep `s` as the final action because it calculates the significance used by the
RL reward. `AnalysisEnv` enforces this for its TREx calls, appending/moving `s`
to the end of its configured action list.

Run the real TREx-backed loop by constructing `AnalysisEnv` with:
```python
AnalysisEnv(
    config_path="configs/hyy.config",
    trex_runner="scripts/trex.py",
    workdir="runs",
    target_significance=5.0,
)
```

Each environment step does:
```text
LLM proposes edit
environment applies edit
TRExFitter runs
environment parses significance
reward = significance - target_significance
```
