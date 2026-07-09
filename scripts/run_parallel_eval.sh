#!/usr/bin/env bash
# Launch MGSM (4-way split) + FLORES in parallel across GPUs 0-3.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BS="${BATCH_SIZE:-4}"

kill_session() { tmux kill-session -t "$1" 2>/dev/null || true; }

kill_session mgsm-eval
kill_session mgsm-g0
kill_session mgsm-g2
kill_session mgsm-g3
kill_session flores-eval

launch_mgsm() {
  local gpu=$1 session=$2 langs=$3 out=$4
  mkdir -p "${ROOT}/${out}"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu BATCH_SIZE=$BS TEST_LANGS='$langs' OUTPUT_DIR='$ROOT/$out' bash scripts/run_mgsm.sh 2>&1 | tee -a '$ROOT/$out/run.log'; echo '--- FINISHED exit:' \$? '---' >> '$ROOT/$out/run.log'"
  echo "[$session] GPU $gpu  langs=$langs  -> $out"
}

launch_mgsm 0 mgsm-g0 "en,bn"     "outputs/mgsm_g0"
launch_mgsm 2 mgsm-g2 "de,es,fr,ja" "outputs/mgsm_g2"
launch_mgsm 3 mgsm-g3 "ru,sw,th,zh" "outputs/mgsm_g3"

mkdir -p "${ROOT}/outputs/flores101"
tmux new-session -d -s flores-eval \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=$BS bash scripts/run_flores.sh 2>&1 | tee -a '$ROOT/outputs/flores101/run.log'; echo '--- FINISHED exit:' \$? '---' >> '$ROOT/outputs/flores101/run.log'"
echo "[flores-eval] GPU 1  langs=all  -> outputs/flores101"

echo ""
echo "All jobs launched. Monitor: tmux ls"
echo "  tmux attach -t mgsm-g0 | mgsm-g2 | mgsm-g3 | flores-eval"
