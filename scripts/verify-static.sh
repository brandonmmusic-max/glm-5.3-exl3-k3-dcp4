#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

bash -n runtime/serve-glm53-exl3-k3-dcp4
python3 -m compileall -q overlays

tail -n +2 evidence/overlay-manifest.tsv | while IFS=$'\t' read -r destination _ expected; do
  [[ -n "$destination" ]] || continue
  actual=$(sha256sum "overlays/$destination" | cut -d' ' -f1)
  [[ "$actual" == "$expected" ]] || {
    echo "hash mismatch: $destination" >&2
    exit 1
  }
done

if rg -n '/home/[^/]+/|/Users/[^/]+/|192\.168\.[0-9]+\.[0-9]+|ghp_[A-Za-z0-9]+' \
    README.md Dockerfile docs evidence publication runtime scripts \
    --glob '!verify-static.sh'; then
  echo "non-portable path, private address, or likely credential found" >&2
  exit 1
fi

echo "static verification passed"
