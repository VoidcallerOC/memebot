# War Room Report — Local Decision Engine

**STATUS: PARTIAL**

No live trading. No keys. No bankroll increase. `$20` ceiling untouched.

---

## CURRENT ARCHITECTURE

```
CHAIN / DexScreener / RPC
    → strategy.py (boosted | smart_money)          [candidate source]
    → meta/ (TokenSnapshot, MetaSignal, clusters) [observational META]
    → safety.py                                   [honeypot / rug screen]
    → risk.py                                     [sizing, stops, daily halt]
    → jupiter.py + main.py                        [quote / dry-run / live]
    → backtest.py                                 [price-path risk replay]
    → preflight.py + process_lock.py              [live gates]
```

**New (this change):**

```
TokenSnapshot + MetaSignal
    → decision/features.py          structured FeatureRecord
    → decision/models/*             local BUY/WATCH/REJECT probability
    → decision/engine.py            typed signal + fail-closed defaults
    → decision/risk_firewall.py     wraps existing safety + risk (model cannot override)
    → decision/shadow.py            latency-adjusted shadow + counterfactuals
```

Frontier LLM is **not** in this path.

---

## EXISTING REUSABLE WORK

| Existing component | Reusable? | Why | Missing capability |
|---|---|---|---|
| META detector (`bot/meta/*`) | YES | Scores, snapshots, history JSONL, clusters | Not a trade decision; no P(target) |
| Token discovery (`strategy.py`) | YES | Boosted + smart_money sources | Not ranked/scored for edge |
| Wallet tracking (`SmartMoneyStrategy`) | PARTIAL | Detects new accumulation | No wallet PnL / win rate / copyability |
| Transaction lifecycle (`jupiter.py`, `main.py`) | YES | Quote, confirm, fill accounting | Do not rebuild |
| Preflight (`preflight.py`) | YES | Fail-closed live eligibility | Do not rebuild |
| Process lock | YES | Single-instance | Do not rebuild |
| Bankroll / `$20` ceiling (`MAX_EXPERIMENT_USD`) | YES | Authoritative live cap | Do not raise |
| Market/price data (DexScreener, Jupiter) | YES | Liquidity, price, windows | No latency-adjusted fill journal |
| Logging / state / alerts | YES | Operational | No decision audit trail |
| Tests | YES | Broad safety/lifecycle coverage | Decision engine tests added |
| Dry-run / paper (`LIVE_TRADING=false`) | PARTIAL | Logs would-be buys at screen price | Not latency-adjusted; no counterfactuals |
| P&L (`portfolio.py`) | YES | Realized/unrealized | Shadow expectancy journal missing historically |
| Feature extraction | PARTIAL | META component scores | No ML feature schema / vector |
| Historical data | PARTIAL | `meta_observations.jsonl` schema | No labeled outcomes in repo |
| Model infrastructure | NO (was missing) | `meta/model.py` is a dataclass module, not ML | Local decision model added |

**DO NOT REDO:** safety, risk, preflight, process lock, Jupiter lifecycle, META formulas, `$20` ceiling.

---

## DECISION TARGET

**Name:** `hit_plus10_before_minus5`

**Definition:** After a **latency-adjusted** entry (price at `detection_ts + latency_seconds`, never the observed wallet fill), does price reach **+10% before −5%** within **3600s**?

| Field | Value |
|---|---|
| Input features | `features.v1` (25 floats from TokenSnapshot + MetaSignal) |
| Label | 1 = TP before stop; 0 = stop, timeout, or no path |
| Horizon | 3600 seconds |
| Default detection latency | 2.5 seconds |
| Known leakage risks | Future META scores; wallet entry price as fill; shuffled train/test |
| Unavailable features | Wallet historical PnL, copyability, SOL/memecoin regime, social velocity, creator graph |
| Outcome definition | See `bot/decision/labels.py` + `schema.py` |

Chosen because the repo can label it from price paths; it does **not** invent a fancier target the data cannot support.

---

## FEATURE SET

Schema version: `features.v1` — see `python -m bot.decision schema`.

Includes: log liquidity/mcap, token age, 5m/1h volume, META velocity scores, buy/sell ratios, tx velocity, holder/creator concentration, liquidity quality, manipulation risk, flow alignment, provider freshness, LP flags, active-meta membership, price changes, missing-feature fraction.

Every feature has source, type, missing behavior, units (in `FEATURE_SPECS`).

---

## MODEL OPTIONS

| Candidate | Why considered | Result on synthetic OOS |
|---|---|---|
| Rules baseline | Existing-threshold heuristics; arm A | prec 0.286 / rec 0.462 (≈ base rate) |
| L2 logistic regression | Low latency, calibrated probs, tiny, CPU, retrainable | prec n/a (0 BUYs at 0.65); ECE **0.014**; fail-closed |
| Stump ensemble | Lightweight non-linear alternative | prec 0.26 / rec 1.0 (predicts everything); ECE 0.49 |
| Neural net | Deferred | Not justified before a real labeled sample |
| Frontier LLM | Arm C | **UNAVAILABLE** in path by design |

**Chosen default artifact:** `logistic.v1` — best calibration, fail-closed under uncertainty, single-digit-µs–class inference. It did **not** earn a precision edge on synthetic labels; it is the safe shadow default until real outcomes exist.

**Simplest model with useful OOS signal:** **none** on the synthetic set (no arm beat base rate + 5pp with BUY volume). Winner selector therefore falls back to logistic.

---

## LATENCY BENCHMARK

Measured locally, `n=2000`, offline (no RPC). Source: `artifacts/decision/latency_benchmark.json`.

| Stage | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| Feature extraction | **0.009 ms** | 0.009 ms | 0.009 ms | 0.012 ms | 0.026 ms |
| Model inference | **0.004 ms** | 0.005 ms | 0.005 ms | 0.006 ms | 0.013 ms |
| End-to-end (infer + risk firewall) | **0.010 ms** | 0.010 ms | 0.011 ms | 0.013 ms | 0.026 ms |

- Throughput: **~42,000 decisions/sec** (offline loop)
- RSS delta: ~0.25 MB
- CPU process time for 2000 iters: ~0.048 s

**Claim:** Local decision path is single-digit **microseconds to ~0.01 ms** end-to-end offline — not the pipeline bottleneck. Network META/RPC collection is outside this measurement and remains the slow part.

Do not treat this as a Jev-equivalent vendor claim.

---

## VALIDATION

Time-aware chronological split (no shuffle):

```
TRAIN  earliest 60%
VALID  next 20%
TEST   latest 20%
```

Implemented in `bot/decision/dataset.py::time_aware_split` and used by `train` / `compare`.

Calibration reported via bins + ECE + Brier (`bot/decision/validate.py`). Confidence ≠ accuracy.

---

## SHADOW RESULTS

Infrastructure: **COMPLETE** (`bot/decision/shadow.py`, `decide-demo` CLI).

Real-chain shadow expectancy: **NOT STARTED** — no labeled historical outcome dataset in the repository.

Demo confirms: wallet entry price is recorded as ignored; simulated fill uses latency-adjusted path price.

---

## PROFITABILITY EVIDENCE

**NO VERIFIED PROFITABILITY.**

Synthetic metrics exist only to prove the train/eval loop. They are not edge.

---

## SECURITY

| Control | Status |
|---|---|
| Dry-run default | Intact |
| `$20` live ceiling | Intact (`MAX_EXPERIMENT_USD`) |
| Preflight / process lock | Untouched |
| Model cannot override risk | Enforced (`RiskFirewall`) |
| Kill switch → BLOCK | Tested |
| Missing/stale/NaN/unavailable model → REJECT | Tested |
| No seed phrases / keys / autonomous signing in this work | Confirmed |
| Live trading enabled by this PR | **NO** |

---

## IMPLEMENTED

- `bot/decision/` package: schema, features, labels, logistic/rules/stumps, engine, risk firewall, shadow journal, time-aware train, validation, benchmark, compare, CLI
- `tests/test_decision_engine.py` (19 tests)
- `docs/WAR_ROOM_DECISION_ENGINE.md` (this report)
- `artifacts/decision/` model JSON + latency + train/comparison summaries
- `numpy` added to `requirements.txt`

## NOT IMPLEMENTED

- Live execution wiring of the decision engine into `main.py` (intentionally gated)
- Frontier LLM arm C offline scores
- Real labeled shadow dataset from chain
- Wallet PnL / copyability features
- Neural nets / GPU inference
- Any bankroll or ceiling change
- Rebuild of META, safety, risk, preflight, lifecycle

## BLOCKERS

1. **No real labeled outcome sample** — cannot claim OOS trading expectancy.
2. **META observation history not present in repo** — need sustained `meta` JSONL + forward prices to label `hit_plus10_before_minus5`.
3. **Arm C unavailable** — no offline LLM score file for fair cost/latency comparison.

## NEXT ACTION

**Run the existing META observational pipeline long enough to persist `market_snapshot` rows, join forward prices with detection latency, emit labeled JSONL, then retrain/evaluate with `python -m bot.decision train --data <that.jsonl>` — still shadow-only.**
