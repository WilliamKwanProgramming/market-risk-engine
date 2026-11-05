#!/usr/bin/env bash
# Regenerate every result in results/validation/ from a clean checkout, then the tables.
# Runs single-threaded; about 15 minutes on one core. Usage: scripts/reproduce.sh
# Needs Python >= 3.12 with numpy, scipy and pandas (the experiments use simulated data only).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PYTHON:-python3}
OUT=results/validation
mkdir -p "$OUT"
rm -f "$OUT"/*.csv "$OUT"/*.manifest.txt "$OUT"/tables.md
start=$(date +%s)
commands=()
run() {
  commands+=("$*")
  echo "+ $*"
  "$PY" "$@"
}
for set in calm typical persistent_fat; do
  run -m mre.experiments.garch_dgp --set "$set" --reps 200 --out "$OUT"
done
run -m mre.experiments.size_power --reps 5000 --out "$OUT"
run -m mre.experiments.basel_zones --windows 200000 --out "$OUT"
run -m mre.experiments.option_tail --batches 20 --batch-size 50000 --out "$OUT"
run scripts/make_tables.py
end=$(date +%s)
"$PY" scripts/write_manifest.py --seconds $((end - start)) --out results/MANIFEST.txt \
  -- "${commands[@]/#/python3 }"
echo "done in $((end - start)) s; see results/MANIFEST.txt"
