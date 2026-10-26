#!/usr/bin/env bash
# Nightly database dump, last 7 kept. Restore: gunzip -c FILE | psql "$DATABASE_URL"
set -euo pipefail
source <(grep '^DATABASE_URL=' /etc/engine.env)
DIR=/home/ubuntu/backups; mkdir -p "$DIR"
pg_dump "$DATABASE_URL" | gzip > "$DIR/engine_$(date +%Y%m%d).sql.gz"
ls -1t "$DIR"/engine_*.sql.gz | tail -n +8 | xargs -r rm --
