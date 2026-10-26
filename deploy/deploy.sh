#!/usr/bin/env bash
# From the Mac: push code (and, the first time, the prepared data) to the VM.
#   deploy/deploy.sh <public-ip>            code only, then restart
#   deploy/deploy.sh <public-ip> --data     also copy the corpus + embeddings and bulk-load the database
# The first run on a fresh VM installs everything (setup_server.sh); later runs skip that.
set -euo pipefail
IP="${1:?usage: deploy.sh <public-ip> [--data]}"
SSH="ssh -o StrictHostKeyChecking=accept-new ubuntu@$IP"
cd "$(dirname "$0")/.."
$SSH "mkdir -p app/data/cache"
rsync -az --delete --exclude .venv --exclude .git --exclude data --exclude '__pycache__' --exclude .env \
      ./ "ubuntu@$IP:app/"
if [ "${2:-}" = "--data" ]; then
  rsync -az data/corpus "ubuntu@$IP:app/data/"
  rsync -az data/cache/structural "ubuntu@$IP:app/data/cache/"
fi
$SSH "test -d app/.venv || sudo bash app/deploy/setup_server.sh $IP"
if [ "${2:-}" = "--data" ]; then
  $SSH "cd app && set -a && . /etc/engine.env && set +a && .venv/bin/python -m engine.store.bulk --strategy structural"
fi
$SSH "sudo systemctl restart engine && sleep 2 && systemctl is-active engine"
