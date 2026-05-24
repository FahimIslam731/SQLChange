#!/usr/bin/env bash
# Demo: run SQLChange recommendation engine against a local Ollama endpoint.
# Usage:
#   ./demo_ollama.sh                  # default: record 0, standard ollama (port 11434)
#   ./demo_ollama.sh 14               # record 14 from dataset
#   ./demo_ollama.sh --caliper        # use Caliper OpenGllama (port 11435)
#   ./demo_ollama.sh --caliper 6      # Caliper + record 6

cd "$(dirname "$0")"

PROVIDER="local"
RECORD=0
MODEL="${OLLAMA_MODEL:-gpt-oss:20b}"

for arg in "$@"; do
    case "$arg" in
        --caliper) PROVIDER="caliper" ;;
        [0-9]*)    RECORD="$arg" ;;
    esac
done

echo "Provider: $PROVIDER | Model: $MODEL | Record: $RECORD"
echo "---"

.venv/bin/python src/recommend_cli.py \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --record "$RECORD" \
    --input data/sqlchange_dataset.json
