# Crypto-Scalping-Bot: FreqAI, Tiers, Risk Manager, and LLM Veto (v1)

This repository contains a professional crypto-scalping trading system built on top of Freqtrade. It integrates FreqAI adaptive modeling, a multi-tier candidate classification scheme, a deterministic Risk Manager, an active LLM Veto context agent, and custom sqlite-based outcome logging and reporting.

---

## 1. Directory Structure (`user_data/`)

All custom system components are organized and modularized within the standard Freqtrade `user_data/` directory, keeping the core engine completely untouched:

```text
user_data/
├── strategies/
│   └── ScalpStrategy.py       # Core strategy with FreqAI feature engineering, tier classification, and risk/LLM hooks
├── ai_layer/
│   ├── __init__.py
│   ├── llm_client.py          # LLM Context Agent client (Groq primary, Gemini fallback, failsafe veto)
│   ├── prompt_templates.py    # Compressed state-to-prompt formatter
│   ├── state_cache.py         # Discretized, per-pair isolated state cache to optimize LLM API usage
│   ├── risk_manager.py        # Deterministic Risk Manager implementing all seven B.6 rules
│   ├── decision_logger.py     # Schema creator and persistent logger for the custom 'ai_decisions' table
│   └── reporting.py           # Core reporting implementation (Weekly review & outcome analysis tool)
├── config_freqai.json         # Master configuration for Freqtrade, FreqAI, Risk Manager, and Tiers
└── tradesv3.dryrun.sqlite     # SQLite database containing native 'trades' and custom 'ai_decisions'
```

---

## 2. Operation Manual

### Running Backtests
Before backtesting, ensure you have downloaded historical candlestick data for the configured coin pairs:
```bash
freqtrade download-data --config user_data/config_freqai.json --timeframes 5m --timerange 20260101-
```
Then, execute the backtest using the ScalpStrategy:
```bash
freqtrade backtesting --config user_data/config_freqai.json --strategy ScalpStrategy --timerange 20260101-
```

### Running Dry-Run Mode
Dry-run operates against real-time market feeds using a virtual wallet. This is the **default and mandatory validation mode** prior to risking real capital.
To launch:
```bash
freqtrade trade --config user_data/config_freqai.json --strategy ScalpStrategy
```
*Note: If booting dry-run for the first time without pre-existing training data, you can temporarily disable FreqAI training under the `"freqai": { "enabled": false }` configuration block to verify connection and strategy hooks.*

### Running the Reporting Tool
`reporting.py` is a manual CLI tool designed for weekly review loops. It joins Freqtrade's native `trades` table with the custom `ai_decisions` table to analyze LGBM predictions, LLM vetoes, and risk checks against actual trade outcomes.

- **To show all historical records (Console Table):**
  ```bash
  python reporting.py
  ```
- **To filter for the last 7 days (Console Table):**
  ```bash
  python reporting.py --last 7days
  ```
- **To filter for the last 14 days and export to CSV:**
  ```bash
  python reporting.py --last 14days --output csv
  ```
  This creates an `ai_decisions_report.csv` file in the repository root.

---

## 3. Tool Comparison & Ecosystem Alignment

There are **three separate and distinct** monitoring and inspection tools in the v1 architecture. Do not conflate them:

| Tool | Purpose | Source of Data | How to Access / Run |
| :--- | :--- | :--- | :--- |
| **Tensorboard** | FreqAI training inspection, loss/accuracy curves, and feature importance across retraining epochs. | FreqAI model training logs (`user_data/models/`) | `tensorboard --logdir user_data/models/` |
| **FreqUI** | High-level trade and wallet metrics, equity curves, win rate, profit factor, and transaction history. | Freqtrade native `trades` table | Enable `api_server` in config; access via browser |
| **reporting.py** | Manual AI decision versus real outcome review (LGBM confidence, LLM veto, risk checks passed/failed vs. Win/Loss). | Left-join of custom `ai_decisions` and native `trades` | `python reporting.py [args]` |

---

## 4. Live Trading Guardrails & Safety Requirements

Transitioning from dry-run to live trading requires strict adherence to safety protocols and system requirements:

1. **The 2-Week Dry-Run Rule (URS §A.6)**:
   Live trading with real capital should **ONLY** be initiated after the system has run in dry-run mode continuously for at least **2 weeks** and demonstrated consistent, stable profitability. No exceptions.
2. **Minimum-Viable-Balance Check (B.6.7)**:
   Live orders will refuse to execute unless the available account balance exceeds the exchange's minimum order notional limit plus a configurable safety buffer (default is minimum notional × 1.5). If this check fails, the trade candidate is blocked, logged as `insufficient_balance`, and bypassed. This protects against system orders getting rejected due to capital depletion.
3. **API Key Security**:
   Ensure your Binance API key and secret are stored securely using environment variables or a separate `.env` file. Never commit credentials to version control. Set exchange permissions to trade-only (withdrawals must be disabled).

---

## 5. Practical Operational Notes

- **Untouched Freqtrade Core**: This project strictly modifies files within `user_data/` (or adds root-level scripts like `reporting.py`). Keep the upstream core engine pristine to ensure seamless package upgrades.
- **SQLite Database Source of Truth**: The custom `ai_decisions` table resides inside Freqtrade's native SQLite database file. This ensures backups, environment separation (e.g. dry-run vs. live database files), and general management remain perfectly aligned.
- **Manual Review Tooling**: `reporting.py` is executed manually as part of your calibration loops. It is not designed to run on a Cron job or scheduling daemon. Use its CSV export feature to maintain offline Excel-compatible logs of model decisions.
