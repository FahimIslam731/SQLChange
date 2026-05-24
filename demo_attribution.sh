#!/usr/bin/env bash
# Demo: run LLM-as-judge attribution analysis.
# Watch Caliper's attention visualizer while this runs to see what tokens
# the model actually attends to.
#
# Usage:
#   ./demo_attribution.sh                    # single record via local ollama
#   ./demo_attribution.sh 14                 # specific record
#   ./demo_attribution.sh --caliper          # use Caliper (port 11435)
#   ./demo_attribution.sh --caliper --batch  # 6 records (one per mutation type)

cd "$(dirname "$0")"

PROVIDER="local"
MODEL="${OLLAMA_MODEL:-qwen3.6:27b}"
RECORD=0
BATCH=false

for arg in "$@"; do
    case "$arg" in
        --caliper) PROVIDER="caliper" ;;
        --batch)   BATCH=true ;;
        [0-9]*)    RECORD="$arg" ;;
    esac
done

echo "Provider: $PROVIDER | Model: $MODEL | Record: $RECORD"
echo "---"

if [ "$BATCH" = true ]; then
    .venv/bin/python src/attribution_cli.py \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --sample-size 6 \
        --input data/sqlchange_dataset.json \
        --output data/attribution_results.json
else
    .venv/bin/python src/attribution_cli.py \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --record "$RECORD" \
        --input data/sqlchange_dataset.json
fi
