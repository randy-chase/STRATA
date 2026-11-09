#!/usr/bin/env bash
set -euo pipefail

# Download all runtime artifacts to local NVMe scratch.
# Run from repository root.

STRATA_NVME_ROOT="/mnt/nvme/strata_v1"
AUX_NVME_ROOT="/mnt/nvme/strata_aux"
HF_CMD=""

mkdir -p "${STRATA_NVME_ROOT}"
mkdir -p "${AUX_NVME_ROOT}"

if command -v hf >/dev/null 2>&1; then
  HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CMD="huggingface-cli"
else
  echo "ERROR: neither 'hf' nor 'huggingface-cli' found in PATH." >&2
  echo "Install HF CLI first, then re-run this script." >&2
  exit 1
fi

# 1) STRATA released model artifacts (includes checkpoints/strata_v1.pth and static fields).
"${HF_CMD}" download nvidia/strata --local-dir "${STRATA_NVME_ROOT}"

# 2) Required vertical coordinate file from this repo.
cp data/scream_vertical_coordinate.nc "${AUX_NVME_ROOT}/"

# 3) Generate auxiliary mesh files including latlon_ne1024pg2.nc.
#    This requires the dependencies listed in data_prep/scrip_generation/requirements.txt.
if [[ -f "data_prep/scrip_generation/generate_all.sh" ]]; then
  bash data_prep/scrip_generation/generate_all.sh --output-dir "${AUX_NVME_ROOT}"
else
  echo "WARNING: data_prep/scrip_generation/generate_all.sh not found in this checkout."
  echo "Please provide ${AUX_NVME_ROOT}/latlon_ne1024pg2.nc manually."
fi

echo "Done."
echo "Set these env vars:"
echo "  STRATA_CHECKPOINT=${STRATA_NVME_ROOT}/checkpoints/strata_v1.pth"
echo "  AUX_DATA_ROOT=${AUX_NVME_ROOT}"

echo ""
echo "Expected files:"
echo "  ${STRATA_NVME_ROOT}/checkpoints/strata_v1.pth"
echo "  ${AUX_NVME_ROOT}/latlon_ne1024pg2.nc"
echo "  ${AUX_NVME_ROOT}/scream_vertical_coordinate.nc"
