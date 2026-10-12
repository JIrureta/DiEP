#!/usr/bin/env bash
# The whole DiEP study in one command: search -> prune -> evaluate -> report,
# queued over the free GPUs.
#
#   ./scripts/run_all.sh                    # GPUS="1 3 5 7" by default
#   GPUS="1 3" ./scripts/run_all.sh         # fewer cards, same jobs, longer
#
# Jobs (the search does not depend on the sparsity, so one per seed serves every r):
#
#   search  seeds 0, 1, 2 -> results/runs/seed{s}/scores.pt
#   diep    global selection, no skipping (the paper's Table 1 setting), every
#           seed x r in {0.25, 0.5, 0.75} -- the headline, mean of 3 seeds
#   uniform same scores, N*r pruned per layer (spec §12's baseline), seed 0
#   merge   pruned experts merged into retained ones (spec §8), seed 0
#   skip    the global model with adaptive skipping (spec §7), seed 0
#   bench   memory ratio and generation throughput (spec §6.1), seed 0
#
# Longest first; each GPU takes the next job whose inputs exist. Restartable:
# a job whose output exists is skipped. The report is rebuilt after every job.
#
# Parameters (environment); the defaults are the original study, results/:
#
#   EPOCHS=10      search epochs over the 32 calibration sequences (2 steps each)
#   METHOD=diep    label in the results; the variants are METHOD-uniform/-merge/-skip
#   ARMS="diep uniform merge skip bench"   which job families to run
#   RESULTS=...    where everything goes. Guarded by RESULTS/study.json: a folder
#                  holding a study with other settings is refused, never mixed --
#                  a search with other epochs would otherwise be "found" and reused
#   PREVIOUS / PREVIOUS_RESULTS / PREVIOUS_LABEL   an earlier DiEP run the report
#                  compares against; PRIMARY_LABEL names this one
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/config.sh"

GPUS="${GPUS:-1 3 5 7}"
EPOCHS="${EPOCHS:-10}"
METHOD="${METHOD:-diep}"
ARMS="${ARMS:-diep uniform merge skip bench}"
PRIMARY_LABEL="${PRIMARY_LABEL:-DiEP (mean of 3 seeds)}"
RUNS="${RESULTS:?}/runs"; HARNESS="${RESULTS:?}/harness"
LOGS="${RESULTS:?}/logs"; QUEUE="${RESULTS:?}/queue"
mkdir -p "$RUNS" "$HARNESS" "$LOGS" "$QUEUE"
[ -f "$CALIBRATION" ] || { echo "calibration not found: $CALIBRATION" >&2; exit 1; }

# One study per folder: refuse a folder that holds different settings.
STUDY="$RESULTS/study.json"
want="{\"epochs\": $EPOCHS, \"method\": \"$METHOD\", \"calibration\": \"$(basename "$CALIBRATION")\"}"
if [ -f "$STUDY" ]; then
    [ "$(cat "$STUDY")" = "$want" ] || {
        echo "refusing: $RESULTS holds a different study" >&2
        echo "  there: $(cat "$STUDY")" >&2; echo "  asked: $want" >&2; exit 1; }
else
    if [ -n "$(ls -A "$RUNS" 2>/dev/null)" ]; then
        echo "refusing: $RUNS has runs but no study.json -- cannot tell which study they are" >&2
        exit 1
    fi
    echo "$want" > "$STUDY"
fi
has_arm() { case " $ARMS " in *" $1 "*) return 0 ;; esac; return 1; }

RATIOS=(0.25 0.5 0.75)                      # sparsity r = fraction REMOVED
JOBS=("search|0|-" "search|1|-" "search|2|-")
has_arm diep && for r in "${RATIOS[@]}"; do JOBS+=("diep|0|$r"); done
for r in "${RATIOS[@]}"; do
    has_arm uniform && JOBS+=("uniform|0|$r")
    has_arm merge && JOBS+=("merge|0|$r")
done
has_arm diep && for s in 1 2; do for r in "${RATIOS[@]}"; do JOBS+=("diep|$s|$r"); done; done
has_arm skip && for r in "${RATIOS[@]}"; do JOBS+=("skip|0|$r"); done
has_arm bench && for r in "${RATIOS[@]}"; do JOBS+=("bench|0|$r"); done

keep_of() { "$PYTHON" -c "print(f'{1 - $1:.3f}')"; }
pruned_dir() { echo "$RUNS/seed$2/pruned-r$3-$1"; }      # variant, seed, r
result_of() {
    local kind="$1" seed="$2" r="$3"
    case "$kind" in
        search)  echo "$RUNS/seed$seed/scores.pt" ;;
        diep)    echo "$HARNESS/$METHOD-keep$(keep_of "$r")-rep$seed-1shot.json" ;;
        uniform) echo "$HARNESS/$METHOD-uniform-keep$(keep_of "$r")-rep$seed-1shot.json" ;;
        merge)   echo "$HARNESS/$METHOD-merge-keep$(keep_of "$r")-rep$seed-1shot.json" ;;
        skip)    echo "$HARNESS/$METHOD-skip-keep$(keep_of "$r")-rep$seed-1shot.json" ;;
        bench)   echo "$RUNS/seed$seed/bench-r$r.json" ;;
    esac
}
ready() {        # are the job's inputs present?
    local kind="$1" seed="$2" r="$3"
    case "$kind" in
        search) return 0 ;;
        diep|uniform|merge) [ -f "$RUNS/seed$seed/scores.pt" ] ;;
        skip)   [ -f "$(pruned_dir global "$seed" "$r")/prune_card.json" ] ;;
        bench)  [ -f "$(pruned_dir global "$seed" "$r")/skip_config.json" ] ;;
    esac
}
blocked_for_good() {   # the producer of an input failed: stop waiting for it
    local kind="$1" seed="$2" r="$3"
    case "$kind" in
        diep|uniform|merge) [ -e "$QUEUE/search-seed$seed.failed" ] ;;
        skip)  [ -e "$QUEUE/diep-seed$seed-r$r.failed" ] ;;
        bench) [ -e "$QUEUE/skip-seed$seed-r$r.failed" ] || [ -e "$QUEUE/diep-seed$seed-r$r.failed" ] ;;
        *) return 1 ;;
    esac
}

prune_once() {   # variant, seed, r, extra args -> a pruned checkpoint (idempotent)
    local variant="$1" seed="$2" r="$3"; shift 3
    local dir; dir="$(pruned_dir "$variant" "$seed" "$r")"
    [ -f "$dir/prune_card.json" ] && return 0
    # The scores must come from this study's search length, nothing else.
    "$PYTHON" -c "import sys, torch; e = torch.load(sys.argv[1], map_location='cpu', weights_only=False)['config']['epochs']; sys.exit(0 if e == int(sys.argv[2]) else f'scores.pt was searched for {e} epochs, this study uses {sys.argv[2]}')" \
        "$RUNS/seed$seed/scores.pt" "$EPOCHS" || return 1
    rm -rf "${dir:?}"              # a half-written checkpoint is redone, never trusted
    local selection="global"; [ "$variant" = uniform ] && selection=uniform
    "$PYTHON" -u -m diep prune --scores "$RUNS/seed$seed/scores.pt" --sparsity "$r" \
        --selection "$selection" --out "$dir" --calibration-file "$CALIBRATION" "$@"
}

eval_pruned() {  # dir, method, extra args
    local dir="$1" method="$2"; shift 2
    "$PYTHON" -u -m diep eval --model "$dir" --method "$method" --tasks all --num-fewshot 1 \
        --allow-unsafe-code --output "$HARNESS" "$@"
}

run_job() {
    local kind="$1" seed="$2" r="$3" gpu="$4"
    export CUDA_VISIBLE_DEVICES="$gpu"
    if [ "${DRY_RUN:-0}" = "1" ]; then           # exercise the queue without a GPU
        echo "DRY $kind seed $seed r $r gpu $gpu"; sleep "${DRY_SECONDS:-2}"
        case "$kind" in
            search) mkdir -p "$RUNS/seed$seed"; : > "$RUNS/seed$seed/scores.pt" ;;
            diep) mkdir -p "$(pruned_dir global "$seed" "$r")"
                  : > "$(pruned_dir global "$seed" "$r")/prune_card.json" ;;
            skip) : > "$(pruned_dir global "$seed" "$r")/skip_config.json" ;;
            bench) echo '{}' > "$(result_of bench "$seed" "$r")" ;;
        esac
        case "$kind" in search|bench) ;; *)
            printf '{"method":"dry","requested_keep_ratio":%s,"mean_mc":0.5,"tasks":{}}' \
                "$(keep_of "$r")" > "$(result_of "$kind" "$seed" "$r")" ;; esac
        return 0
    fi
    case "$kind" in
    search)
        "$PYTHON" -u -m diep search --model "$MODEL" --calibration-file "$CALIBRATION" \
            --out "$RUNS/seed$seed" --seed "$seed" --epochs "$EPOCHS" ;;
    diep)
        prune_once global "$seed" "$r" || return 1
        eval_pruned "$(pruned_dir global "$seed" "$r")" "$METHOD" ;;
    uniform)
        prune_once uniform "$seed" "$r" || return 1
        eval_pruned "$(pruned_dir uniform "$seed" "$r")" "$METHOD-uniform" ;;
    merge)
        prune_once merge "$seed" "$r" --merge || return 1
        eval_pruned "$(pruned_dir merge "$seed" "$r")" "$METHOD-merge" ;;
    skip)
        local dir; dir="$(pruned_dir global "$seed" "$r")"
        if [ ! -f "$dir/skip_config.json" ]; then
            "$PYTHON" -u -m diep calib-skip --model "$dir" --calibration-file "$CALIBRATION" \
                --out "$dir/skip_config.json.tmp" || return 1
            mv "$dir/skip_config.json.tmp" "$dir/skip_config.json"
        fi
        eval_pruned "$dir" "$METHOD-skip" --skip "$dir/skip_config.json" ;;
    bench)
        local dir; dir="$(pruned_dir global "$seed" "$r")"
        "$PYTHON" -u -m diep bench --model "$dir" --skip "$dir/skip_config.json" \
            --baseline "$MODEL" --calibration-file "$CALIBRATION" \
            --out "$(result_of bench "$seed" "$r")" ;;
    esac
}

report() {
    [ -f "$REPO_ROOT/examples/report.py" ] || return 0
    ( flock 9
      previous=()
      [ -n "${PREVIOUS:-}" ] && previous=(--previous "$PREVIOUS" --previous-label "$PREVIOUS_LABEL"
                                          --previous-results "$PREVIOUS_RESULTS")
      "$PYTHON" "$REPO_ROOT/examples/report.py" --results "$RESULTS" --ga-harness "$GA_HARNESS" \
          --dpma-results "$DPMA_RESULTS" --primary "$METHOD" --primary-label "$PRIMARY_LABEL" \
          "${previous[@]}" > "$LOGS/report.log" 2>&1 \
          || echo "  (report failed -- see $LOGS/report.log)"
    ) 9> "$RESULTS/.report.lock"
}

worker() {
    local gpu="$1" job kind seed r name ran pending
    while true; do
        ran=0; pending=0
        for job in "${JOBS[@]}"; do
            IFS='|' read -r kind seed r <<< "$job"
            name="$kind-seed$seed$( [ "$r" = - ] || echo "-r$r")"
            [ -e "$(result_of "$kind" "$seed" "$r")" ] && continue
            [ -e "$QUEUE/$name.claim" ] && continue
            if ! ready "$kind" "$seed" "$r"; then
                blocked_for_good "$kind" "$seed" "$r" || pending=1
                continue
            fi
            mkdir "$QUEUE/$name.claim" 2>/dev/null || continue     # atomic: one GPU per job
            echo "$(date -u +%H:%M) [gpu $gpu] > $name"
            # In a subshell: an error inside a job fails that job, not this worker.
            if ( run_job "$kind" "$seed" "$r" "$gpu" ) > "$LOGS/$name.log" 2>&1 \
                    && [ -e "$(result_of "$kind" "$seed" "$r")" ]; then
                echo "$(date -u +%H:%M) [gpu $gpu] . $name done"
            else
                echo "$(date -u +%H:%M) [gpu $gpu] ! $name FAILED (see $LOGS/$name.log)"
                : > "$QUEUE/$name.failed"
            fi
            report
            ran=1; break                     # rescan from the top: priority order
        done
        [ "$ran" -eq 1 ] && continue
        [ "$pending" -eq 1 ] && { sleep "${POLL:-60}"; continue; }
        return 0
    done
}

banner "DiEP [$METHOD, $EPOCHS epochs] on $MODEL | ${#JOBS[@]} jobs over GPUs: $GPUS"
echo "  calibration: $CALIBRATION"
echo "  results:     $RESULTS"
# A restart re-queues anything unfinished.
find "${QUEUE:?}" -maxdepth 1 \( -name '*.claim' -o -name '*.failed' \) -exec rm -rf {} +
started=$(date +%s)
pids=()
for gpu in $GPUS; do worker "$gpu" & pids+=($!); sleep "${STAGGER:-15}"; done
for pid in "${pids[@]}"; do wait "$pid"; done
report
missing=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r kind seed r <<< "$job"
    if [ ! -e "$(result_of "$kind" "$seed" "$r")" ]; then
        missing=$((missing + 1)); echo "  missing: $kind seed $seed r $r"
    fi
done
banner "finished in $(( ($(date +%s) - started) / 60 )) min: $(( ${#JOBS[@]} - missing ))/${#JOBS[@]} jobs have results"
echo "  analysis  $RESULTS/ANALYSIS.md"
echo "  figures   $RESULTS/figures/"
[ "$missing" -eq 0 ]
