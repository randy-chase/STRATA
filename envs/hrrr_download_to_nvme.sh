#!/usr/bin/env bash
set -euo pipefail

# Download all runtime artifacts to local NVMe scratch.
# Run from repository root.

mkdir -p /mnt/nvme/strata_v1
mkdir -p /mnt/nvme/strata_aux

# 1) STRATA released model artifacts (includes checkpoints/strata_v1.pth and static fields).
huggingface-cli download nvidia/strata --local-dir /mnt/nvme/strata_v1

# 2) Required vertical coordinate file from this repo.
cp data/scream_vertical_coordinate.nc /mnt/nvme/strata_aux/

# 3) Required lat/lon file:
#    place latlon_ne1024pg2.nc in /mnt/nvme/strata_aux/
#    If you already have it, copy it now:
# cp /path/to/latlon_ne1024pg2.nc /mnt/nvme/strata_aux/

echo "Done."
echo "Set these env vars:"
echo "  STRATA_CHECKPOINT=/mnt/nvme/strata_v1/checkpoints/strata_v1.pth"
echo "  AUX_DATA_ROOT=/mnt/nvme/strata_aux"
