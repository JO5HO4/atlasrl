
# Setup

## RL Agent Environemnt
```bash
conda env create -f setup/environment.yml
conda activate atlasrl
```

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

Run histogram creation. The wrapper automatically splits the `n` step by
region and uses one worker per region job by default:
```bash
python scripts/trex.py configs/hyy.config --actions n
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
