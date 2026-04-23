#!/bin/zsh
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="$BASE_DIR/runtime-operations-config/runtime-config.txt"

while IFS='=' read -r key value; do
  [[ -z "${key:-}" || "$key" == \#* ]] && continue
  export "$key=$value"
done < "$CONFIG_FILE"

cd "$WORKDIR"
exec "$CONDA_BIN" run -n "$CONDA_ENV" python "$BASE_DIR/$PYTHON_ENTRY"
