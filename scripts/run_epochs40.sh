#!/usr/bin/env bash
# DiEP with the paper's number of search steps, in one command.
#
#   ./scripts/run_epochs40.sh              # GPUS="1 3 5 7" by default
#
# The original study (results/) ran the spec's 10 epochs, which over the 32
# calibration sequences is 20 steps (15 alpha / 5 beta); the paper's 128
# sequences make the same 10 epochs 80 steps. This runs 40 epochs = 80 steps
# (60 alpha / 20 beta) over the SAME calibration file, into its own folder with
# its own label, so nothing of the original study is touched:
#
#   results/epochs40/          runs, harness, figures, ANALYSIS.md, card.md
#   label diep-80steps         (the original is diep)
#
# Arms: plain DiEP at 3 seeds x 3 ratios, and seed-0 uniform and merge. The
# report compares against the original 20-step DiEP as well as the GA and DPMA.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$here/.." && pwd)"
export RESULTS="$REPO_ROOT/results/epochs40"
export EPOCHS=40 METHOD=diep-80steps ARMS="diep uniform merge"
export PRIMARY_LABEL="DiEP, 80 steps (mean of 3 seeds)"
export PREVIOUS=diep PREVIOUS_RESULTS="$REPO_ROOT/results"
export PREVIOUS_LABEL="DiEP, 20 steps (mean of 3 seeds)"
exec "$here/run_all.sh" "$@"
