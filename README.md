
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
  -v "$PWD":/workdir \
  -w /workdir \
  gitlab-registry.cern.ch/atlas/statanalysis:0-4
```