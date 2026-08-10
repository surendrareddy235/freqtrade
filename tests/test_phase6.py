"""
Unit and integration tests for Phase 6 (LLM veto activation) components:
Verifies that ScalpStrategy's confirm_trade_entry correctly processes the actual
llm_veto value and successfully blocks/allows candidates based on that veto.
"""
import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd

from user_data.strategies.ScalpStrategy import ScalpStrategy
from freqtrade.persistence import Trade

@pytest.fixture
def test_config(tmp_path):
    db_file = os.path.join(tmp_path, "test_trades.sqlite")
    return {
        "dry_run": True,
        "max_open_trades": 1,
        "stake_currency": "USDT",
        "dry_run_wallet": 1000.0,
        "db_url": f"sqlite:///{db_file}",
        "risk_manager": {
            "max_daily_trades": 10,
            "daily_drawdown_limit": 0.05,
            "position_size_percent": 1.0,
            "correlation_threshold": 0.85,
            "min_balance_buffer": 1.5,
        },
        "tier_classification": {
            "tier_1_threshold": 0.80,
            "tier_2_threshold": 0.60,
            "tier_3_threshold": 0.55
        },
        "freqai": {
            "enabled": False,
        }
    }

@patch("freqtrade.persistence.Trade.get_trades_proxy")
def test_confirm_trade_entry_veto_active(mock_get_trades, test_config):
    # Mock no open trades initially
    mock_get_trades.return_value = []

    # Instantiate strategy
    strategy = ScalpStrategy(test_config)
    strategy.dp = MagicMock()
    strategy.dp.current_whitelist.return_value = ["DOGE/USDT"]

    # Pre-populate close returns cache to bypass get_analyzed_dataframe
    strategy.close_returns_cache["DOGE/USDT"] = pd.Series([0.001] * 30)

    # 1. Simulate Tier 2 candidate with LLM veto = True
    strategy.latest_candidate_data["DOGE/USDT"] = {
        "timestamp": datetime.now(),
        "tier": 2,
        "model_confidence": 0.75,
        "di_ok": 1,
        "llm_invoked": 1,
        "llm_provider": "groq",
        "llm_veto": 1,  # True
        "llm_confidence": 0.9,
        "llm_reason": "news check failed"
    }

    # Execute confirm_trade_entry
    result = strategy.confirm_trade_entry(
        pair="DOGE/USDT",
        order_type="limit",
        amount=100.0,
        rate=0.1,
        time_in_force="gtc",
        current_time=datetime.now(),
        entry_tag="long_scalp",
        side="buy"
    )

    # Veto is True, so trade MUST be blocked
    assert result is False

    # Check database record is logged as blocked by llm_veto
    db_file = test_config["db_url"].replace("sqlite:///", "")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT tier, llm_veto, risk_checks_passed, block_reason, outcome FROM ai_decisions WHERE pair='DOGE/USDT'")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 2  # Tier 2
    assert row[1] == 1  # llm_veto = True
    import json
    breakdown = json.loads(row[2])
    assert breakdown["llm_veto"] is False
    assert row[3] == "llm_veto"
    assert row[4] == "rejected"


@patch("freqtrade.persistence.Trade.get_trades_proxy")
def test_confirm_trade_entry_veto_inactive(mock_get_trades, test_config):
    # Mock no open trades initially
    mock_get_trades.return_value = []

    # Instantiate strategy
    strategy = ScalpStrategy(test_config)
    strategy.dp = MagicMock()
    strategy.dp.current_whitelist.return_value = ["DOGE/USDT"]

    # Pre-populate close returns cache to bypass get_analyzed_dataframe
    strategy.close_returns_cache["DOGE/USDT"] = pd.Series([0.001] * 30)

    # 2. Simulate Tier 2 candidate with LLM veto = False
    strategy.latest_candidate_data["DOGE/USDT"] = {
        "timestamp": datetime.now(),
        "tier": 2,
        "model_confidence": 0.75,
        "di_ok": 1,
        "llm_invoked": 1,
        "llm_provider": "groq",
        "llm_veto": 0,  # False
        "llm_confidence": 0.9,
        "llm_reason": "news check passed"
    }

    # Execute confirm_trade_entry
    result = strategy.confirm_trade_entry(
        pair="DOGE/USDT",
        order_type="limit",
        amount=100.0,
        rate=0.1,
        time_in_force="gtc",
        current_time=datetime.now(),
        entry_tag="long_scalp",
        side="buy"
    )

    # Veto is False, so trade should pass Risk Manager rules
    assert result is True

    # Check database record is logged as passed/approved
    db_file = test_config["db_url"].replace("sqlite:///", "")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT tier, llm_veto, risk_checks_passed, block_reason, outcome FROM ai_decisions WHERE pair='DOGE/USDT'")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 2  # Tier 2
    assert row[1] == 0  # llm_veto = False
    import json
    breakdown = json.loads(row[2])
    assert all(breakdown.values()) is True
    assert row[3] is None
    assert row[4] == "approved"
