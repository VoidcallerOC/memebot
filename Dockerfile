# Solana memecoin bot — container image.
# Runs the trading loop; DRY-RUN by default (LIVE_TRADING must be set in .env).
FROM python:3.11-slim

# Don't buffer stdout/stderr so `docker logs` shows ticks in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STATE_FILE=/data/state.json

WORKDIR /app

# Install deps first so the layer caches across code changes.
# Runtime dependencies come from the canonical hash-locked set that CI also
# tests: requirements/ci-py3.11.lock (CPython 3.11, linux x86_64,
# manylinux_2_28 wheels). The lock pins versions and hashes every published
# file of each version (all platforms), so the target is enforced explicitly:
# the build aborts unless the base image is CPython 3.11 / x86_64 /
# glibc >= 2.28, and also on a mismatched hash, an sdist, a broken dependency
# graph (pip check), or a requirements.txt entry the lock does not satisfy.
# Nothing is resolved or upgraded at build time.
COPY requirements.txt ./
COPY requirements/ci-py3.11.lock requirements/ci-py3.11.lock
RUN python -c "import platform, sys; l = platform.libc_ver(); ok = sys.version_info[:2] == (3, 11) and platform.machine() == 'x86_64' and l[0] == 'glibc' and tuple(int(x) for x in l[1].split('.')[:2]) >= (2, 28); sys.exit(0 if ok else 'unsupported runtime target: %s %s %s' % (sys.version.split()[0], platform.machine(), l))" \\
    && python -m pip install --no-cache-dir --disable-pip-version-check \
        --require-hashes --no-deps --only-binary :all: -r requirements/ci-py3.11.lock \
    && python -m pip check \
    && python -m pip install --no-cache-dir --disable-pip-version-check \
        --no-index --dry-run -r requirements.txt

COPY bot/ ./bot/

# Persist state (open positions + daily circuit breaker) on a volume, and run
# as a non-root user that owns it.
RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /data \
    && chown -R bot:bot /data /app
USER bot
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "bot.main"]
