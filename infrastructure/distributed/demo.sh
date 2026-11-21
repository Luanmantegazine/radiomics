#!/usr/bin/env bash
# Distributed demo: coordinator + dashboard + three site containers (synthetic data).
# Needs Docker and python3. Tokens are generated into ./.secrets (never committed).
# BUILD_FLAG=--no-build reuses images that are already built.
set -euo pipefail
cd "$(dirname "$0")"
: "${POSTGRES_PASSWORD:=demo-only-change-me}"
export POSTGRES_PASSWORD
COMPOSE=(docker compose -f ../docker-compose.yml -f docker-compose.sites.yml)
API=http://localhost:8000

mkdir -p .secrets
python3 - <<'PY'
import hashlib, secrets, pathlib
lines = ["sites:"]
for n in ("01", "02", "03"):
    token = secrets.token_urlsafe(32)
    path = pathlib.Path(f".secrets/client-{n}.token")
    path.write_text(token)
    path.chmod(0o644)  # read by the container user
    lines += [f"  - client_id: client-{n}",
              f"    token_sha256: {hashlib.sha256(token.encode()).hexdigest()}"]
pathlib.Path(".secrets/sites.yaml").write_text("\n".join(lines) + "\n")
PY

export EXPERIMENT_ID=pending
"${COMPOSE[@]}" up -d ${BUILD_FLAG---build} postgres mlflow backend frontend
until curl -fsS "$API/health" >/dev/null; do sleep 2; done

# YAML -> JSON inside the backend container (no Python packages needed on the host).
CONFIG_JSON=$("${COMPOSE[@]}" exec -T backend python -c \
  "import json, sys, yaml; print(json.dumps(yaml.safe_load(sys.stdin)))" < experiment.yaml)
EXPERIMENT_ID=$(curl -fsS -X POST "$API/experiments" -H 'content-type: application/json' \
  -d "$CONFIG_JSON" | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
export EXPERIMENT_ID
echo "experiment: $EXPERIMENT_ID"
curl -fsS -X POST "$API/experiments/$EXPERIMENT_ID/run" >/dev/null
"${COMPOSE[@]}" --profile sites up -d site-01 site-02 site-03

while true; do
  status=$(curl -fsS "$API/experiments/$EXPERIMENT_ID" | python3 -c "import json,sys;print(json.load(sys.stdin)['status'])")
  [[ "$status" == completed || "$status" == failed ]] && break
  sleep 3
done
echo "status: $status"
echo "dashboard: http://localhost:8080/experiments/$EXPERIMENT_ID"
[[ "$status" == completed ]]
