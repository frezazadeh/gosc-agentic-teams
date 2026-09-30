#!/bin/sh
# Decompress the large per-mission result files (kept gzip-compressed in the
# repository to stay below GitHub's file-size limit).
cd "$(dirname "$0")/../results" && gunzip -kf noise.jsonl.gz overhead.jsonl.gz shared.jsonl.gz
echo "results/*.jsonl unpacked"
