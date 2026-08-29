#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}
data_home=${XDG_DATA_HOME:-"${HOME:?HOME is required}/.local/share"}
config_home=${XDG_CONFIG_HOME:-"${HOME:?HOME is required}/.config"}
vendor_root=${ROBOFORGE_VENDOR_ROOT:-"$data_home/roboforge/vendor"}
vendor_config=${ROBOFORGE_LIBERO_VENDOR_CONFIG:-"$config_home/roboforge/libero_vendor.json"}

"$python_bin" - <<'PY'
import sys
if sys.version_info >= (3, 12):
    raise SystemExit("full LIBERO deployment requires Python 3.10 or 3.11")
PY

clone_at() {
    local url=$1
    local revision=$2
    local destination=$3
    if [[ -e "$destination" && ! -d "$destination/.git" ]]; then
        echo "refusing unmanaged source directory: $destination" >&2
        exit 2
    fi
    if [[ ! -d "$destination/.git" ]]; then
        git clone --filter=blob:none "$url" "$destination"
    fi
    git -C "$destination" fetch --depth 1 origin "$revision"
    git -C "$destination" checkout --detach "$revision"
    test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
}

mkdir -p "$vendor_root"
"$python_bin" -m pip install -e "$repo_root[libero]"

clone_at https://github.com/Lifelong-Robot-Learning/LIBERO.git \
    8f1084e3132a39270c3a13ebe37270a43ece2a01 "$vendor_root/libero"
clone_at https://github.com/IDEA-Research/GroundingDINO.git \
    856dde20aee659246248e20734ef9ba5214f5e44 "$vendor_root/GroundingDINO"
clone_at https://github.com/facebookresearch/segment-anything.git \
    dca509fe793f601edb92606367a655c15ac00fdf "$vendor_root/segment-anything"
clone_at https://github.com/graspnet/graspnet-baseline.git \
    280c215129f759ed8649cb4e89fc5dfee55f4f80 "$vendor_root/graspnet-baseline"

# LIBERO's namespace layout is omitted by both modern editable mappings and its
# regular wheel. Setuptools' compat mode adds the pinned source root to sys.path.
"$python_bin" -m pip install --no-deps --force-reinstall -e "$vendor_root/libero" \
    --config-settings editable_mode=compat

groundingdino_patch="$repo_root/scripts/patches/groundingdino-pytorch2.patch"
if git -C "$vendor_root/GroundingDINO" apply --check "$groundingdino_patch"; then
    git -C "$vendor_root/GroundingDINO" apply "$groundingdino_patch"
elif ! git -C "$vendor_root/GroundingDINO" apply --reverse --check "$groundingdino_patch"; then
    echo "GroundingDINO compatibility patch does not apply to the pinned revision" >&2
    exit 2
fi

"$python_bin" -m pip install --no-build-isolation -e "$vendor_root/GroundingDINO"
"$python_bin" -m pip install -e "$vendor_root/segment-anything"
"$python_bin" -m pip install -r "$vendor_root/graspnet-baseline/requirements.txt"
mkdir -p "$vendor_root/graspnet-baseline/knn/knn_pytorch"
touch "$vendor_root/graspnet-baseline/knn/knn_pytorch/__init__.py"
"$python_bin" -m pip install --no-build-isolation -e "$vendor_root/graspnet-baseline/knn"
mkdir -p "$vendor_root/graspnet-baseline/pointnet2/pointnet2"
touch "$vendor_root/graspnet-baseline/pointnet2/pointnet2/__init__.py"
"$python_bin" -m pip install --no-build-isolation -e "$vendor_root/graspnet-baseline/pointnet2"

"$python_bin" "$repo_root/scripts/download_libero_text_encoder.py" \
    --destination "$vendor_root/bert-base-uncased"

mkdir -p "$(dirname "$vendor_config")"
"$python_bin" - "$vendor_config" "$vendor_root" <<'PY'
import json
import os
from pathlib import Path
import sys

target = Path(sys.argv[1]).expanduser().resolve()
vendor = Path(sys.argv[2]).expanduser().resolve()
payload = {"protocol": "roboforge-libero-vendor-v1", "sources": {
    "groundingdino": str(vendor / "GroundingDINO"),
    "groundingdino_text_encoder": str(vendor / "bert-base-uncased"),
    "segment_anything": str(vendor / "segment-anything"),
    "graspnet": str(vendor / "graspnet-baseline"),
}}
temporary = target.with_suffix(target.suffix + ".tmp")
with temporary.open("w") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, target)
PY

cat <<EOF
Pinned LIBERO source and optional perception sources installed. The package uses
the fixed LIBERO Git revision above rather than the upstream archive, whose
metadata is not consistently installable by pip.
ROBOFORGE_GROUNDINGDINO_ROOT=$vendor_root/GroundingDINO
ROBOFORGE_GROUNDINGDINO_TEXT_ENCODER=$vendor_root/bert-base-uncased
ROBOFORGE_SAM_ROOT=$vendor_root/segment-anything
ROBOFORGE_GRASPNET_ROOT=$vendor_root/graspnet-baseline
ROBOFORGE_LIBERO_VENDOR_CONFIG=$vendor_config
EOF
