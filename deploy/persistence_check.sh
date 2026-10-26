#!/usr/bin/env bash
# Proves the brief's "data persists after an application restart": upload a probe, restart the
# service, check the counts and that search still finds it, then delete the probe.
# Run on the server: bash app/deploy/persistence_check.sh
set -euo pipefail
set -a; . /etc/engine.env; set +a
B=http://127.0.0.1:8000
health() { curl -s $B/api/health | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["documents"], d["chunks"])'; }
printf "# Persistence probe\n\nProbe record PERSIST-7731 written before a restart.\n" > /tmp/persist_probe.md
curl -sf -H "X-Admin-Token: $ADMIN_TOKEN" -F file=@/tmp/persist_probe.md $B/api/documents >/dev/null
before=$(health)
sudo systemctl restart engine
for i in $(seq 1 90); do curl -s $B/api/health | grep -q '"ok"' && break; sleep 2; done
after=$(health)
found=$(curl -s -X POST $B/api/search -H "Content-Type: application/json" -d '{"query":"PERSIST-7731","k":1}' \
        | python3 -c 'import json,sys; h=json.load(sys.stdin); print(bool(h) and h[0]["doc_id"]=="uploads/persist_probe.md")')
curl -sf -X DELETE -H "X-Admin-Token: $ADMIN_TOKEN" "$B/api/documents?source=uploads/persist_probe.md" >/dev/null
echo "before restart: $before | after restart: $after | probe found after restart: $found"
[ "$before" = "$after" ] && [ "$found" = "True" ] && echo PASS || { echo FAIL; exit 1; }
