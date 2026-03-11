# nobrainer-runner

Job dispatch backend for [nobrainer](https://github.com/neuronets/nobrainer).
Supports Slurm (HPC clusters), AWS Batch, and GCP Batch.

## Installation

```bash
uv venv --python 3.11
uv pip install -e .
# With AWS support:
uv pip install -e ".[aws]"
# With GCP support:
uv pip install -e ".[gcp]"
```

## Usage

```bash
nobrainer-runner submit --backend slurm --config job.yaml
nobrainer-runner status --job-id <id>
```

## Backends

- **Slurm**: HPC cluster job submission via `sbatch`
- **AWS Batch**: Cloud GPU jobs on AWS
- **GCP Batch**: Cloud GPU jobs on Google Cloud
