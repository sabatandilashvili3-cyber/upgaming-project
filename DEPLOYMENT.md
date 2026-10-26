# Deployment

## Architecture

```
 browser ──HTTPS──► Caddy :443 ──► uvicorn :8000 (FastAPI, systemd "engine") ──► Postgres 16 + pgvector
                  (auto certificate,        │  in-memory BM25 index,                (same VM, local disk)
                   <ip>.sslip.io)           │  bge-small embedder (CPU)
                                            └──HTTPS──► Anthropic API (claude-opus-5-5)
```

Everything runs on **one Oracle Cloud Always Free VM**: Ampere A1 (ARM), 4 OCPU, 24 GB RAM,
Ubuntu 24.04, 200 GB block storage. No paid infrastructure; the only spend is the LLM API.

**Why this host:** I checked three options against the brief before choosing.
- **Hugging Face Spaces (rejected):** Docker Spaces now require a paid plan, and a Space can
  only open outbound connections on ports 80, 443 and 8080, so it couldn't reach Postgres on
  5432 anyway.
- **Render / Koyeb free (rejected):** 512 MB RAM. The app measures about 550 MB:
  - 130 MB of chunks in memory;
  - 215 MB of BM25 index;
  - 130 MB of PyTorch embedding model.
- **Oracle (chosen):** 24 GB RAM, ordinary networking, and a disk that survives restarts.

| Requirement | How it's met |
|---|---|
| Public URL | `https://<ip-with-dashes>.sslip.io`. sslip.io resolves any such name to the IP, so Caddy can get a Let's Encrypt certificate without buying a domain |
| Hosting + database integration | The app connects to local Postgres through a connection pool that checks each connection before use |
| Data survives restarts | Postgres stores data on the VM's boot volume. The app is stateless: it rebuilds BM25 from the database at start (about 20 s) |
| Secrets protected | They live in `/etc/engine.env` (mode 600, never in git), loaded by systemd. `.env` is git-ignored locally |
| Abuse and cost control | Per-client hourly limit, a daily LLM budget enforced from the `traces` table, and an admin token for uploads and deletes |
| Backups | Nightly `pg_dump`, the last 7 kept in `/home/ubuntu/backups` |

**Database size:** 480 MB for 61,911 records:
- chunks + vectors: 160 MB;
- HNSW index: 125 MB;
- LSH bands: 108 MB;
- documents: 59 MB.

That's a small fraction of the 200 GB volume.

**Free tier limits and how they show up:**
- **Oracle may reclaim an Always Free VM that sits idle** (about 20% CPU or less for 7 days).
  Real use keeps it busy; the nightly backup adds a little.
- **Ampere capacity is sometimes "out of capacity"** in popular regions. Retry, pick another
  availability domain, or use a smaller shape (2 OCPU / 12 GB is plenty).
- **No sleeping or cold start**, unlike Render or Spaces: the VM is always on.
- **LLM cost** is the real limit. One investigation costs roughly $0.05–0.30. `DAILY_BUDGET_USD`
  (default 5) stops new investigations for the day once it's reached.

## Step by step

### 1. Create the VM (Oracle console, about 15 minutes)
1. Sign up at cloud.oracle.com. A card is required for identity checks; Always Free resources
   are not charged. Pick a home region close to you; it can't be changed later.
2. **Compute → Instances → Create instance:**
   - **Image:** Canonical Ubuntu 24.04.
   - **Shape:** Ampere → `VM.Standard.A1.Flex`, 4 OCPU, 24 GB (or 2 / 12).
   - **Networking:** keep "Create new virtual cloud network" with a public subnet, and
     "Assign a public IPv4 address".
   - **SSH keys:** paste your public key (`cat ~/.ssh/id_ed25519.pub`).
   - Create the instance, and copy its **Public IP address**.
3. **Open ports 80 and 443:**
   1. Go to the instance → its subnet → **Security list** → **Add ingress rules**.
   2. Add two rules, each with source `0.0.0.0/0`, protocol TCP, destination port `80` for one
      and `443` for the other.

   The VM's own firewall is opened by the setup script.

### 2. Deploy (from the Mac, in the project folder)
```bash
ssh ubuntu@<IP> 'echo ok'                 # first login; accept the host key
deploy/deploy.sh <IP> --data              # code + data, installs everything, loads the database
```
The first run takes about 10–15 minutes (packages, PyTorch, the bulk load).

### 3. Add the LLM key on the server (never in git)
```bash
ssh ubuntu@<IP>
sudo nano /etc/engine.env                 # set ANTHROPIC_API_KEY=...
sudo systemctl restart engine
grep ADMIN_TOKEN /etc/engine.env          # needed for uploads/deletes in the web page
```

### 4. Check
```bash
curl https://<ip-with-dashes>.sslip.io/api/health
```
Then open the URL and ask the March question.

### Later deploys
```bash
deploy/deploy.sh <IP>                     # code only: rsync + restart, a few seconds
```

### Useful commands on the server
```bash
journalctl -u engine -f                   # app logs
sudo systemctl restart engine
sudo -u postgres psql -d engine           # database
ls ~/backups                              # nightly dumps; restore: gunzip -c FILE | psql "$DATABASE_URL"
```
