#!/usr/bin/env bash
# Shared settings for the DiEP study. Sourced by the other scripts.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-allenai/OLMoE-1B-7B-0924}"
#: The one calibration file every method in this study uses (GA, DPMA, DiEP).
CALIBRATION="$REPO_ROOT/../GA/MoE_compression_GA/scripts/results/calibration/c4-32x1024-seed0.pt"
PYTHON="${PYTHON:-/home/jupyter/workspace/llm_evaluation/llm-eval/bin/python}"
RESULTS="${RESULTS:-$REPO_ROOT/results}"
GA_HARNESS="$REPO_ROOT/../GA/MoE_compression_GA/results/harness"
DPMA_RESULTS="$REPO_ROOT/../DPMA/results"
export DS_IGNORE_CUDA_DETECTION=1                       # no nvcc in this container
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
banner() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
