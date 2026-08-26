#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}
data_home=${XDG_DATA_HOME:-"${HOME:?HOME is required}/.local/share"}
config_home=${XDG_CONFIG_HOME:-"${HOME:?HOME is required}/.config"}
vendor_root=${ROBOFORGE_VENDOR_ROOT:-"$data_home/roboforge/vendor"}
vendor_config=${ROBOFORGE_LIBERO_VENDOR_CONFIG:-"$config_home/roboforge/libero_vendor.json"}

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

clone_at https://github.com/IDEA-Research/GroundingDINO.git \
    856dde20aee659246248e20734ef9ba5214f5e44 "$vendor_root/GroundingDINO"
clone_at https://github.com/facebookresearch/segment-anything.git \
    dca509fe793f601edb92606367a655c15ac00fdf "$vendor_root/segment-anything"
clone_at https://github.com/graspnet/graspnet-baseline.git \
    280c215129f759ed8649cb4e89fc5dfee55f4f80 "$vendor_root/graspnet-baseline"

"$python_bin" -m pip install --no-build-isolation -e "$vendor_root/GroundingDINO"
"$python_bin" -m pip install -e "$vendor_root/segment-anything"
"$python_bin" -m pip install -r "$vendor_root/graspnet-baseline/requirements.txt"
mkdir -p "$vendor_root/graspnet-baseline/knn/knn_pytorch"
touch "$vendor_root/graspnet-baseline/knn/knn_pytorch/__init__.py"
"$python_bin" -m pip install --no-build-isolation -e "$vendor_root/graspnet-baseline/knn"
mkdir -p "$vendor_root/graspnet-baseline/pointnet2/pointnet2"
touch "$vendor_root/graspnet-baseline/pointnet2/pointnet2/__init__.py"
"$python_bin" -m pip install --no-build-isolation -e "$vendor_root/graspnet-baseline/pointnet2"

"$python_bin" - "$vendor_root/bert-base-uncased" <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

destination = Path(sys.argv[1]).resolve()
snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    revision="86b5e0934494bd15c9632b12f734a8a67f723594",
    local_dir=str(destination),
    allow_patterns=["config.json", "pytorch_model.bin", "tokenizer.json",
                    "tokenizer_config.json", "vocab.txt"],
)
PY

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
