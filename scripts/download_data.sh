#!/usr/bin/env bash
# Downloads MRAG-Bench's QA split. Requires huggingface.co network access.
# The 16,130-image retrieval corpus is a separate archive from the MRAG-Bench
# authors (see the printed instructions from download_mrag_bench.py) -- once
# you have it, re-run with --image_archive path/to/archive.zip
set -euo pipefail

python data/download_mrag_bench.py --out_dir data/mrag_bench "$@"
