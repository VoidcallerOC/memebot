# War Room Report — Local Decision Engine

**STATUS: PARTIAL**

No live trading. No keys. No bankroll increase. `$20` ceiling untouched.

---

## PHASE STATUS (0–13)

| Phase | Status | Evidence |
|---|---|---|
| 0 Audit | COMPLETE | Map below; existing META/safety/risk/preflight/lifecycle reused |
| 1 Decision target | COMPLETE | `hit_plus10_before_minus5` documented in `schema.py` |
| 2 Feature pipeline | COMPLETE | `features.v1` (25 floats) with source/missing/units |
| 3 Local model | COMPLETE | Rules / logistic / stumps trained + compared; default `logistic.v1` |
| 4 Speed benchmark | COMPLETE | p50/p90/p95/p99 measured (`latency_benchmark.json`) |
| 5 Calibration | PARTIAL | ECE/Brier/bins/confusion on synthetic OOS; **not** on real labels |
| 6 Time-aware validation | COMPLETE | Chronological split + rolling walk-forward folds |
| 7 Shadow trading | PARTIAL | Dry-run bridge + journal; no real-chain expectancy yet |
| 8 Counterfactuals | PARTIAL | Rejects journaled with MFE/MAE/outcome fields; needs live candidate stream |
| 9 Risk firewall | COMPLETE | Independent BLOCK; model cannot override; wraps existing safety/risk |
| 10 Performance comparison | PARTIAL | A/B + walk-forward + shadow PnL/drawdown/opportunity cost; Arm C UNAVAILABLE; synthetic only |
| 11 No live money | COMPLETE | Shadow gates dry-run only; armed path not model-gated; checklist unmet for `$20` |
| 12 Model versioning | COMPLETE | All four version fields on every decision/journal row |
| 13 Failure modes | PARTIAL | Missing/stale/NaN/model/corrupt/unknown token/kill/duplicate/halt tested; RPC/restart-during-shadow still rely on existing bot layers |

---

## CURRENT ARCHITECTURE

```
CHAIN / DexScreener / RPC
    → strategy.py (boosted | smart_money)
    → meta/ (TokenSnapshot, MetaSignal, clusters)     [reused]
    → safety.py + risk.py                             [reused; firewall wraps]
    → decision/ (features → local model → BUY/WATCH/REJECT)
    → decision/risk_firewall.py                       [deterministic veto]
    → decision/shadow journal (dry-run when DECISION_SHADOW=true)
    → jupiter.py + main.py                            [unchanged live gates]
```

Frontier LLM is **not** in this path.

---

## EXISTING REUSABLE WORK

| Existing component | Reusable? | Why | Missing capability |
|---|---|---|---|
| META detector | YES | Scores, snapshots, history | Not a trade decision |
| Token discovery | YES | Boosted + smart_money | Not ranked for edge |
| Wallet tracking | PARTIAL | New-mint accumulation | No wallet PnL / copyability |
| Tx lifecycle / preflight / lock | YES | Do not rebuild | — |
| `$20` ceiling | YES | Authoritative | Do not raise |
| Dry-run paper | PARTIAL | Now optional decision gate | Latency-adjusted fills need price path |
| Historical META JSONL schema | YES | Labeling via `label-meta` | No committed labeled outcomes |
| Model infra | was NO | Added `bot/decision` | Real labels still absent |

**DO NOT REDO:** safety, risk, preflight, process lock, Jupiter lifecycle, META formulas, `$20` ceiling.

---

## DECISION TARGET

**`hit_plus10_before_minus5`**

After latency-adjusted entry (`detection_ts + 2.5s`), does price reach **+10% before −5%** within **3600s**?

- Entry never uses observed wallet fill price
- Leakage risks: future META scores, wallet fill as entry, shuffled splits
- Unavailable: wallet PnL history, copyability, SOL/memecoin regime, social velocity

---

## FEATURE SET

`features.v1` — 25 floats from TokenSnapshot + MetaSignal (liquidity, mcap, age, volumes, META velocities, buy/sell ratios, holders, manipulation, flow alignment, freshness, LP flags, active-meta, price changes, missing fraction).  
See `python -m bot.decision schema`.

---

## MODEL OPTIONS

| Candidate | Role | Synthetic OOS note |
|---|---|---|
| Rules | Arm A baseline | prec ≈ base rate |
| L2 logistic | Default artifact | Best ECE (~0.014); fail-closed at 0.65 |
| Stump ensemble | Arm B alt | Over-buys; worse calibration |
| Neural net | Deferred | Not justified |
| Frontier LLM | Arm C | **UNAVAILABLE** |

No model beat base rate + 5pp on synthetic data → default **logistic.v1** (calibrated, fail-closed), not a profitability claim.

---

## LATENCY BENCHMARK

Offline, n=2000 (see `artifacts/decision/latency_benchmark.json`):

| Stage | p50 | p99 | max |
|---|---|---|---|
| Feature extraction | ~0.008–0.009 ms | ~0.011 ms | ~0.027 ms |
| Model inference | ~0.004 ms | ~0.006 ms | ~0.013 ms |
| End-to-end (+ risk) | ~0.009–0.010 ms | ~0.013 ms | ~0.026 ms |

Throughput ~40k+ decisions/sec. Local model is not the pipeline bottleneck. Network META/RPC is outside this measurement.

---

## VALIDATION

1. Chronological 60/20/20 split (`time_aware_split`)
2. Rolling walk-forward folds (`walk_forward_folds`) used in `compare`
3. Metrics: accuracy, precision, recall, confusion, FPR/FNR, ECE, Brier, illustrative EV, shadow expectancy/drawdown/opportunity cost

Confidence ≠ accuracy.

---

## SHADOW RESULTS

| Item | Status |
|---|---|
| Journal schema + latency-adjusted entry | COMPLETE |
| Counterfactual rejects | COMPLETE (infra) |
| Dry-run wiring (`DECISION_SHADOW=true`) | COMPLETE |
| Real-chain shadow expectancy | NOT STARTED |

Enable: `DECISION_SHADOW=true` (dry-run only). Live armed path is **not** model-gated.

---

## PROFITABILITY EVIDENCE

**NO VERIFIED PROFITABILITY.**

Synthetic shadow expectancy exists for plumbing checks only.

---

## SECURITY

| Control | Status |
|---|---|
| Dry-run default | Intact |
| `$20` live ceiling | Intact |
| Preflight / process lock | Untouched |
| Model cannot override risk | Enforced |
| Decision gate on live | **Disabled by design** (Phase 11) |
| Failure → NO TRADE | Default |
| Keys / seeds / bankroll hikes in this work | None |

---

## IMPLEMENTED

- `bot/decision/` — schema, features, labels, models, engine, risk firewall, shadow, shadow_sim, bridge, label_meta, train, validate, benchmark, compare, CLI
- Dry-run hook in `main.py` behind `DECISION_SHADOW` (opt-in)
- Walk-forward + richer A/B comparison
- Tests: `test_decision_engine.py`, `test_decision_phases.py` (161 passed suite-wide)
- Artifacts under `artifacts/decision/`
- Config knobs in `config.example.env`

## NOT IMPLEMENTED

- Live model gating / `$20` experiment enablement
- Frontier LLM arm C scores
- Real labeled dataset from chain
- Wallet PnL / copyability features
- Neural nets

## BLOCKERS

1. No real labeled outcome sample (META snapshots without forward price paths cannot label)
2. Arm C unavailable (by design until offline score file exists)
3. Shadow expectancy on real candidates not yet accumulated

## NEXT ACTION

**Run META observational pipeline until multiple `market_snapshot` rows per mint exist, then `python -m bot.decision label-meta --observations meta_observations.jsonl` and `python -m bot.decision train --data artifacts/decision/labeled_from_meta.jsonl` — still shadow-only.**
