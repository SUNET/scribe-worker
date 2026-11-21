#!/usr/bin/env bash
# Run the transcription evaluation (scripts/eval.py) on this machine.
#
#   scripts/eval.sh                  prepare the data if it is missing, then run
#   scripts/eval.sh run --splits dev quick run while tuning
#   scripts/eval.sh promote          accept the last run as the baseline
#   scripts/eval.sh <anything>       passed on to eval.py (see its --help)
#
# Environment:
#   EVAL_DATA      data directory                  (default: ./eval-data)
#   EVAL_LANGUAGE  language to evaluate            (default: swedish)
#   EVAL_NOISE     music or room recording to mix in, used when preparing
#   EVAL_REAL      folder of recordings with corrected NAME.txt files
#
# It reads the worker's .env like the worker does, so it evaluates the models
# and settings that production uses. Stop the worker first (or run it on a
# machine without jobs): a busy GPU makes the speed numbers meaningless.
set -euo pipefail

cd "$(dirname "$0")/.."

DATA="${EVAL_DATA:-eval-data}"
LANGUAGE="${EVAL_LANGUAGE:-swedish}"

if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
else
    PYTHON=(python)
fi

eval_py() {
    "${PYTHON[@]}" scripts/eval.py "$1" --data "$DATA" --language "$LANGUAGE" "${@:2}"
}

prepare() {
    local extra=()
    [ -n "${EVAL_NOISE:-}" ] && extra+=(--noise "$EVAL_NOISE")
    [ -n "${EVAL_REAL:-}" ] && extra+=(--real "$EVAL_REAL")
    eval_py prepare "${extra[@]}"
}

if [ $# -eq 0 ]; then
    [ -f "$DATA/$LANGUAGE/dev/set.json" ] || prepare
    eval_py run
elif [ "$1" = "prepare" ]; then
    shift
    extra=()
    [ -n "${EVAL_NOISE:-}" ] && extra+=(--noise "$EVAL_NOISE")
    [ -n "${EVAL_REAL:-}" ] && extra+=(--real "$EVAL_REAL")
    eval_py prepare "${extra[@]}" "$@"
else
    eval_py "$@"
fi
