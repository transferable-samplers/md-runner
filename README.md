# MD Runner

**MD Runner** is the molecular-dynamics toolkit used to generate the **ManyPeptidesMD** dataset introduced in our paper:
[**Amortized Sampling with Transferable Normalizing Flows**](https://arxiv.org/abs/2508.18175)

The codebase builds on and extends the MD simulation provided in [**TimeWarp**](https://github.com/microsoft/timewarp).

## Installation

```bash
micromamba env create -f environment.yaml
micromamba activate md-runner
```

## Workflow

### 1. Generate Initial PDB Structures from Sequences

We use **tLEaP** (AmberTools) to construct peptide initial conformations from amino-acid sequences.

To convert a set of sequences into PDB files:

```bash
python src/seq_to_pdb.py seq_filename=sequences/example_sequences.txt
```

### 2. Run Molecular Dynamics Simulations (Local)

You can perform a local test/benchmark MD simulation using:

```bash
python src/generate_md.py seq_name=AA
```

### 3. Run Molecular Dynamics Simulations (SLURM)

For large-scale dataset generation, use the [hydra-submitit-launcher](https://hydra.cc/docs/plugins/submitit_launcher/) plug-in. An example script is provide in:

```bash
./submitit/run-md.py
```

Each individual sequence in `seq_filename` will be launched as a separate SLURM job.
