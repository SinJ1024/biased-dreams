#!/usr/bin/env bash
# Offline cached-prior comparison; validates paths and invokes the Python CLI.
# Template only: not submitted. Supply account, partition, time, memory, and
# CPU/GPU resource requests as site-approved sbatch options before submission.
# Activate the correct module/conda environment here or before submitting.
# Required exports: REPO_DIR (repository root), OUTPUT_DIR (new result directory).
# Optional exports: PYTHON_BIN, DEVICE, RUN_DIR. No cluster paths are invented.
#SBATCH --job-name=compare-cached-uq
#SBATCH --output=compare-cached-uq-%j.log

set -euo pipefail
: "${REPO_DIR:?Set REPO_DIR to the existing biased-dreams checkout}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new directory outside the source run}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
RUN_DIR="${RUN_DIR:-/scratch/jingyuansun/biased-dreams/uncertainty-aware-dreamer/out/paper-core-pd/li-urssm/cheetah_run/0}"

cd "${REPO_DIR}/uncertainty-aware-dreamer"
for name in config.yaml networks.pth id_prior_infos.pt ood_prior_infos.pt random_rollouts.pt; do
    test -f "${RUN_DIR}/${name}" || { echo "Missing source: ${RUN_DIR}/${name}" >&2; exit 1; }
done

srun "$PYTHON_BIN" compare_uncertainty.py \
    --run-dir "$RUN_DIR" \
    --infos "$RUN_DIR/id_prior_infos.pt" "$RUN_DIR/ood_prior_infos.pt" "$RUN_DIR/random_rollouts.pt" \
    --metrics gjs js transition_var --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" --batch-size 256 \
    --js-num-samples 128 --js-chunk-size 8 --js-seed 2026 --js-repeats 5 \
    --dimensions 0 1 --plot-trajectory 0 --surface
