#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [heavy]" >&2
  echo "Run normally with no argument, or use 'heavy' for higher queue load." >&2
}

if (( $# > 1 )); then
  usage
  exit 2
fi

mode=${1:-default}
if [[ "${mode}" != "default" && "${mode}" != "heavy" ]]; then
  usage
  exit 2
fi

DATASETS="intas tapas roma tdrive"
INCLUDE_GRU=${INCLUDE_GRU:-0}
SERVER_COUNT=${SERVER_COUNT:-9}
PLACEMENT=${PLACEMENT:-density}
MAX_STEPS=${MAX_STEPS:-1500}
EVALUATION_WINDOWS=${EVALUATION_WINDOWS:-12}

if [[ "${mode}" == "heavy" ]]; then
  ARRIVAL_RATE=${ARRIVAL_RATE:-4.0}
  CAPACITY_SCALE=${CAPACITY_SCALE:-0.75}
  EVALUATION_WINDOW_SECONDS=${EVALUATION_WINDOW_SECONDS:-300}
  ENVIRONMENT_DIR=${ENVIRONMENT_DIR:-configs/environments/heavy}
  SAVED_MODELS=${SAVED_MODELS:-saved_models_heavy}
  RESULTS_DIR=${RESULTS_DIR:-results/heavy}
else
  ARRIVAL_RATE=${ARRIVAL_RATE:-0.5}
  CAPACITY_SCALE=${CAPACITY_SCALE:-1.0}
  EVALUATION_WINDOW_SECONDS=${EVALUATION_WINDOW_SECONDS:-600}
  ENVIRONMENT_DIR=${ENVIRONMENT_DIR:-configs/environments}
  SAVED_MODELS=${SAVED_MODELS:-saved_models}
  RESULTS_DIR=${RESULTS_DIR:-results}
fi

base_algorithms=(
  greedy
  random
  uniform_random
  kalman
  markov
  coverage_load
)

mkdir -p "${ENVIRONMENT_DIR}" "${RESULTS_DIR}/plots"

for dataset in ${DATASETS}; do
  environment_file="${ENVIRONMENT_DIR}/${dataset}.json"
  algorithms=("${base_algorithms[@]}")
  evaluation_sampling=()

  if [[ "${dataset}" == "intas" || "${dataset}" == "tapas" ]]; then
    algorithms+=(furthest)
  fi
  if [[ "${INCLUDE_GRU}" == "1" ]]; then
    algorithms+=(gru)
  fi

  if [[ "${dataset}" == "roma" || "${dataset}" == "tdrive" ]]; then
    evaluation_sampling=(
      --evaluation-windows "${EVALUATION_WINDOWS}"
      --evaluation-window-seconds "${EVALUATION_WINDOW_SECONDS}"
    )
  fi

  echo "Creating ${mode} environment for ${dataset}..."
  edge-project place-servers \
    --dataset "${dataset}" \
    --servers "${SERVER_COUNT}" \
    --placement "${PLACEMENT}" \
    --resource-profile eco balanced accelerated \
    --capacity-scale "${CAPACITY_SCALE}" \
    --output "${environment_file}"

  echo "Plotting ${dataset}..."
  edge-project plot \
    --dataset "${dataset}" \
    --environment-file "${environment_file}" \
    --max-steps "${MAX_STEPS}" \
    --output "${RESULTS_DIR}/plots/${dataset}_mobility.png"

  echo "Training ${dataset} agents: ${algorithms[*]}"
  edge-project train \
    --dataset "${dataset}" \
    --environment-file "${environment_file}" \
    --saved-models "${SAVED_MODELS}" \
    --algorithm "${algorithms[@]}"

  echo "Evaluating ${dataset} agents at arrival rate ${ARRIVAL_RATE}..."
  edge-project evaluate \
    --dataset "${dataset}" \
    --environment-file "${environment_file}" \
    --saved-models "${SAVED_MODELS}" \
    --algorithm "${algorithms[@]}" \
    --arrival-rate "${ARRIVAL_RATE}" \
    "${evaluation_sampling[@]}" \
    --output "${RESULTS_DIR}/evaluation_${dataset}"
done

echo "All ${mode} simulations completed successfully."
