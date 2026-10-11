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

The image installs its Python dependencies from the hash-locked
`requirements/ci-py3.11.lock` (the same file CI tests on Python 3.11), not
from `requirements.txt`. The lock only contains linux **x86_64** wheels, so
the build refuses (fails closed) on an arm64 host (e.g. Raspberry Pi, Apple
Silicon) unless you build for `linux/amd64`.

## Option B — systemd (plain VM, no Docker)

**Precondition:** Linux **x86_64**, glibc **>= 2.28**, **CPython 3.11**
(`python3.11 -VV` and `ldd --version` to check). The runtime dependencies are
installed from the hash-locked `requirements/ci-py3.11.lock`, which was
resolved for exactly that target (manylinux_2_28 x86_64 wheels). The lock
hashes every published file of each pinned version, so it does **not** by
itself reject another platform: the guard command below does, and the install
must stop if it fails. Do not fall back to resolving `requirements.txt` on the
host. The target VM is not pinned by this repo and has not been validated;
confirm it meets the precondition first.

See the install steps in the header of [`memebot.service`](./memebot.service).
Summary:

```bash
sudo git clone https://github.com/VoidcallerOC/memebot.git /opt/memebot
cd /opt/memebot
sudo python3.11 -m venv .venv
# Refuse anything but CPython 3.11 / x86_64 / glibc >= 2.28:
sudo .venv/bin/python -c "import platform, sys; l = platform.libc_ver(); ok = sys.version_info[:2] == (3, 11) and platform.machine() == 'x86_64' and l[0] == 'glibc' and tuple(int(x) for x in l[1].split('.')[:2]) >= (2, 28); sys.exit(0 if ok else 'unsupported runtime target: %s %s %s' % (sys.version.split()[0], platform.machine(), l))"
sudo .venv/bin/python -m pip install --require-hashes --no-deps --only-binary :all: \
    -r requirements/ci-py3.11.lock
sudo .venv/bin/python -m pip check
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
