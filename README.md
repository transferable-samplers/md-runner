# MD Runner Tool

This codebase is the toolkit used to generate the ManyPeptidesMD dataset, in our work Amortized Sampling with Transferable Normalizing Flows.

The codebase was adapted from the TimeWarp [codebase](https://github.com/microsoft/timewarp)

## Install

```bash
micromamba env create -f environment.yaml
micromamba activate md-runner
```

## Steps

### 1. Generate PDB Files from Sequences

We use `tLEaP` via `ambertools` to generate PDB files with initial conformations. Run the `seq_to_pdb.py` script to convert sequences into PDB files:

```bash
python python src/seq_to_pdb.py seq_filename=sequences/example_sequences.txt
```

### 2. Run Molecular Dynamics Simulations (Local)

Test / benchmark the MD simulation with the following command:

```bash
python src/generate_md.py pdb_filename=AA
```

### 2. Run Molecular Dynamics Simulations (SLURM)

Use the generated PDB files to run molecular dynamics simulations with `generate_md.py`. Copy and string-ify the sequences from the sequences.txt and list them off in the sequences arg in `scripts/run-md.py`.

```bash
./scripts/run-md.py
```