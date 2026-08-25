#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}
vendor_root=${ROBOFORGE_VENDOR_ROOT:-"$repo_root/third_party"}

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

"$python_bin" -m pip install -e "$vendor_root/GroundingDINO"
"$python_bin" -m pip install -e "$vendor_root/segment-anything"
"$python_bin" -m pip install -r "$vendor_root/graspnet-baseline/requirements.txt"
"$python_bin" -m pip install -e "$vendor_root/graspnet-baseline/knn"
"$python_bin" -m pip install -e "$vendor_root/graspnet-baseline/pointnet2"

cat <<EOF
Pinned LIBERO sources installed.
ROBOFORGE_GROUNDINGDINO_ROOT=$vendor_root/GroundingDINO
ROBOFORGE_SAM_ROOT=$vendor_root/segment-anything
ROBOFORGE_GRASPNET_ROOT=$vendor_root/graspnet-baseline
EOF
