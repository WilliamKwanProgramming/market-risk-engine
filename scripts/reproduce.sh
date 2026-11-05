#!/usr/bin/env bash
# Regenerate every result in results/validation/ (simulated data) and results/realdata/ (real
# data: skfolio S&P 500 panel 1990-2022), then the tables. Runs single-threaded.
# Usage: scripts/reproduce.sh            (everything)
#        scripts/reproduce.sh validation (simulated data only)
#        scripts/reproduce.sh realdata   (real data only; needs pip install skfolio arch)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}
PART=${1:-all}
start=$(date +%s)
commands=()
run() {
  commands+=("$*")
  echo "+ $*"
  "$PY" "$@"
}
if [[ $PART == all || $PART == validation ]]; then
  OUT=results/validation
  mkdir -p "$OUT"
  rm -f "$OUT"/*.csv "$OUT"/*.manifest.txt "$OUT"/tables.md
  for set in calm typical persistent_fat; do
    run -m mre.experiments.garch_dgp --set "$set" --reps 200 --out "$OUT"
  done
  run -m mre.experiments.size_power --reps 5000 --out "$OUT"
  run -m mre.experiments.basel_zones --windows 200000 --out "$OUT"
  run -m mre.experiments.option_tail --batches 20 --batch-size 50000 --out "$OUT"
  for set in calm typical persistent_fat; do
    run -m mre.experiments.hs_regime --set "$set" --reps 20 --out "$OUT"
  done
  for T in 250 1000; do
    run -m mre.experiments.es_power --T "$T" --reps 200 --out "$OUT"
  done
fi
if [[ $PART == all || $PART == realdata ]]; then
  OUT=results/realdata
  mkdir -p "$OUT"
  rm -rf "$OUT"/*.csv "$OUT"/*.manifest.txt "$OUT"/tables.md "$OUT"/stage1
  run -m mre.experiments.realdata --checks --out "$OUT"
  for p in EQ20 SPX STRANGLE; do
    run -m mre.experiments.realdata --portfolio "$p" --out "$OUT"
  done
  run -m mre.experiments.realdata --summarise --out "$OUT"
fi
run scripts/make_tables.py
end=$(date +%s)
# user + system CPU time of all child processes (second line of the bash builtin "times", which
# must run in this shell, not in a pipeline)
timesfile=$(mktemp)
times > "$timesfile"
cpu=$(tail -1 "$timesfile" | awk '{s = 0; for (i = 1; i <= 2; i++) {split($i, a, "m");
  sub("s", "", a[2]); s += a[1] * 60 + a[2]}; printf "%.0f", s}')
rm -f "$timesfile"
"$PY" scripts/write_manifest.py --seconds $((end - start)) --cpu-seconds "$cpu" --part "$PART" \
  --out results/MANIFEST.txt -- "${commands[@]/#/python3 }"
echo "done in $((end - start)) s wall, $cpu s CPU of child processes; see results/MANIFEST.txt"
