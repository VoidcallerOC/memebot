# Solana memecoin bot — container image.
# Runs the trading loop; DRY-RUN by default (LIVE_TRADING must be set in .env).
FROM python:3.11-slim

# Don't buffer stdout/stderr so `docker logs` shows ticks in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STATE_FILE=/data/state.json

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/

# Persist state (open positions + daily circuit breaker) on a volume, and run
# as a non-root user that owns it.
RUN useradd --create-home --uid 10001 bot \
    && mkdir -p /data \
    && chown -R bot:bot /data /app
USER bot
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "bot.main"]
