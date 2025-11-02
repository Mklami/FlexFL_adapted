#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-bugs.tsv}"
BASE_DIR="prepare/buggy_program/Collect_Methods/repos"

while read -r proj id kind; do
  # skip header, comments, and blank lines
  [[ "$proj" == "project" ]] && continue
  [[ "$proj" =~ ^#|^$ ]] && continue
  case "$kind" in
    b) v="${id}b"; suffix="buggy" ;;
    f) v="${id}f"; suffix="fixed" ;;
    *) echo "Unknown kind '$kind' (use b/f) for $proj-$id" >&2; exit 1 ;;
  esac
  out="$BASE_DIR/${proj}-${id}_${suffix}"
  if [[ ! -d "$out" ]]; then
    echo "Checking out $proj-$v → $out"
    defects4j checkout -p "$proj" -v "$v" -w "$out"
  else
    echo "Exists: $out (skip)"
  fi
done < "$MANIFEST"
