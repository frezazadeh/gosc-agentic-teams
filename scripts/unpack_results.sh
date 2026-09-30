#!/bin/sh
# Decompress the large per-mission result files of the second revision
# (kept gzip-compressed in the repository to stay below GitHub's file-size limit).
cd "$(dirname "$0")/../results_r2" && gunzip -kf noise.jsonl.gz overhead.jsonl.gz shared.jsonl.gz
echo "results_r2/*.jsonl unpacked"
