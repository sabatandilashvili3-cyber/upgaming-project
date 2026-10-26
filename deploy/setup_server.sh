#!/usr/bin/env bash
# One-time (and safe to re-run) setup of an Oracle Cloud Always Free VM:
# Ubuntu 24.04 on Ampere A1 (ARM). Run as the default "ubuntu" user:
#   sudo bash ~/app/deploy/setup_server.sh <public-ip>
# Installs Postgres 16 + pgvector, Python deps, Caddy (HTTPS), the systemd service,
# the nightly backup, and opens ports 80/443 in the VM's own firewall.
set -euo pipefail
IP="${1:?usage: setup_server.sh <public-ip>}"
HOST="${IP//./-}.sslip.io"            # free hostname that resolves to the IP, so Caddy can get a certificate
APP=/home/ubuntu/app
ENV_FILE=/etc/engine.env

echo "== packages"
apt-get update -qq
apt-get install -y -qq curl gnupg ca-certificates python3-venv python3-dev build-essential \
    debian-keyring debian-archive-keyring apt-transport-https netfilter-persistent iptables-persistent

# Postgres 16 + pgvector from the official PostgreSQL apt repository
if ! dpkg -s postgresql-16-pgvector >/dev/null 2>&1; then
  install -d /usr/share/postgresql-common/pgdg
  curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc
  echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt noble-pgdg main" \
    > /etc/apt/sources.list.d/pgdg.list
  apt-get update -qq
  apt-get install -y -qq postgresql-16 postgresql-16-pgvector
fi

# Caddy from its official repository
if ! command -v caddy >/dev/null; then
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy
fi

echo "== database"
if [ ! -f "$ENV_FILE" ]; then
  DBPASS=$(openssl rand -hex 24)
  sudo -u postgres psql -qc "create role engine login password '$DBPASS'" 2>/dev/null || true
  sudo -u postgres psql -qc "create database engine owner engine" 2>/dev/null || true
  sudo -u postgres psql -qd engine -c "create extension if not exists vector"
  cat > "$ENV_FILE" <<EOF
DATABASE_URL=postgresql://engine:$DBPASS@localhost:5432/engine
ANTHROPIC_API_KEY=
LLM_EFFORT=high
RATE_PER_HOUR=20
DAILY_BUDGET_USD=5
ADMIN_TOKEN=$(openssl rand -hex 16)
EOF
  chmod 600 "$ENV_FILE"; chown ubuntu:ubuntu "$ENV_FILE"
  echo "   wrote $ENV_FILE (database password and admin token generated; add ANTHROPIC_API_KEY there)"
fi

echo "== python environment"
sudo -u ubuntu bash -c "
  cd $APP
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  # CPU-only PyTorch: the default Linux wheel pulls CUDA libraries this machine can't use
  .venv/bin/pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  grep -v -E '^(torch|nvidia-|triton|cuda-)' requirements.txt > /tmp/req.txt
  .venv/bin/pip install -q -r /tmp/req.txt
  .venv/bin/python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')\"
"

echo "== firewall (Oracle's Ubuntu image blocks everything but SSH in iptables)"
for port in 80 443; do
  if ! iptables -C INPUT -p tcp --dport $port -j ACCEPT 2>/dev/null; then
    # insert above the image's final REJECT rule; rules after it are never reached
    at=$(iptables -L INPUT --line-numbers -n | awk '$2 == "REJECT" {print $1; exit}')
    if [ -n "$at" ]; then iptables -I INPUT "$at" -p tcp --dport $port -j ACCEPT
    else iptables -A INPUT -p tcp --dport $port -j ACCEPT; fi
  fi
done
netfilter-persistent save >/dev/null

echo "== service, proxy, backup"
install -m 644 "$APP/deploy/engine.service" /etc/systemd/system/engine.service
sed "s/{HOST}/$HOST/" "$APP/deploy/Caddyfile" > /etc/caddy/Caddyfile
install -m 755 "$APP/deploy/backup.sh" /usr/local/bin/engine-backup
echo "15 3 * * * ubuntu /usr/local/bin/engine-backup" > /etc/cron.d/engine-backup
systemctl daemon-reload
systemctl enable --now engine
systemctl restart caddy


echo "== done: https://$HOST  (also open ports 80 and 443 in the VCN security list, see DEPLOYMENT.md)"
