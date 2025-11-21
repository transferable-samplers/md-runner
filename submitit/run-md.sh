#!/bin/bash
python src/generate_md.py -m launcher=example \
seq_filename=sequences/example_sequences.txt \
seq_idx="range(0, 3)"
