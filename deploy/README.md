# Deploying the bot

The live loop has to run 24/7 to trade, so it belongs on an always-on host (a
small cloud VM, a VPS, or a Raspberry Pi) rather than a laptop. Two supported
ways to run it, both restart on crash and reboot. **Start in dry-run**
(`LIVE_TRADING=false`) until you trust the behavior.

Everything the bot needs is outbound-only — there are no inbound ports and no
web UI. You watch it through logs.

## Option A — Docker (recommended)

Portable, isolated, one command. On any host with Docker:

```bash
git clone https://github.com/VoidcallerOC/memebot.git
cd memebot
cp config.example.env .env      # then edit .env (see below)
docker compose up -d --build    # build + run in the background
docker compose logs -f          # watch it tick; Ctrl-C just stops watching
```

Manage it:

```bash
docker compose restart          # after editing .env
docker compose down             # stop (state persists in the named volume)
docker compose pull && docker compose up -d --build   # after a git pull
```

Open positions and the daily circuit-breaker state live in the `memebot-state`
volume, so restarts and rebuilds don't orphan a live position or reset your
daily loss limit.

## Option B — systemd (plain VM, no Docker)

See the install steps in the header of [`memebot.service`](./memebot.service).
Summary:

```bash
sudo git clone https://github.com/VoidcallerOC/memebot.git /opt/memebot
cd /opt/memebot
sudo python3 -m venv .venv && sudo .venv/bin/pip install -r requirements.txt
sudo cp config.example.env .env && sudoedit .env
sudo useradd --system --home /opt/memebot memebot
sudo chown -R memebot:memebot /opt/memebot
sudo cp deploy/memebot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now memebot
journalctl -u memebot -f
```

## Configuring `.env`

Minimum for a safe dry-run: just `cp config.example.env .env` and go — it
runs locked to `LIVE_TRADING=false`.

Before going live you must set **both** (this double-lock is intentional):

```ini
LIVE_TRADING=true
WALLET_PRIVATE_KEY=<base58 key of a DEDICATED burner wallet>
RPC_URL=<a private RPC endpoint — the public one is rate-limited>
```

Fund the wallet with only what you can afford to lose. Never commit `.env` —
it is git-ignored and must never be baked into the image.

## Security notes

- The image runs as a non-root user and bakes in **no** secrets; `.env` is
  supplied only at runtime.
- Keep `.env` readable only by you: `chmod 600 .env`.
- A leaked `WALLET_PRIVATE_KEY` is a drained wallet. Rotate it if in doubt.
