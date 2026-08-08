"""
Unit tests for Phase 5 (LLM Context Agent) components:
- LLMClient
- StateCache
- PromptTemplates
- DecisionLogger
- Reporting
"""
import os
import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from user_data.ai_layer.llm_client import LLMClient
from user_data.ai_layer.state_cache import StateCache
from user_data.ai_layer.prompt_templates import SYSTEM_PROMPT, build_compressed_payload
from user_data.ai_layer.decision_logger import DecisionLogger
from user_data.ai_layer.reporting import main as reporting_main

def test_prompt_templates_payload():
    indicators = {
        "rsi-14": 55.4,
        "natr-14": 0.01234,
        "relative-volume": 1.23,
        "volume_anomaly": True,
        "volatility_anomaly": False,
        "high_atr": True
    }
    payload = build_compressed_payload("DOGE/USDT", "5m", indicators, 0.76)

    # Verify expected tokens/fields are in the compressed string
    assert "TICK:DOGE" in payload
    assert "TF:5m" in payload
    assert "LGBM:0.76" in payload
    assert "RSI:55" in payload
    assert "NATR:0.0123" in payload
    assert "VREL:1.2" in payload
    assert "V_ANOM:1" in payload
    assert "VOLT_ANOM:0" in payload
    assert "HATR:1" in payload

def test_state_cache():
    cache = StateCache()

    ind1 = {
        "close": 1.23456,
        "rsi-14": 55.4,
        "natr-14": 0.01234,
        "relative-volume": 1.23,
        "volatility_state": 0.42,  # rounds to 0.4
        "model_prediction": 1,
        "model_confidence": 0.762,
        "do_predict": 1,
        "di_value": 0.12,
        "volume_anomaly": True,
        "volatility_anomaly": False,
        "high_atr": True
    }

    # A tiny floating point shift that rounds to the same bin
    ind2 = {
        "close": 1.23459,
        "rsi-14": 55.1,  # rounds to same bin (55/2)*2 vs (55.4/2)*2 -> 56 vs 56
        "natr-14": 0.01231,
        "relative-volume": 1.21, # rounds to 1.2
        "volatility_state": 0.44, # rounds to 0.4
        "model_prediction": 1,
        "model_confidence": 0.758, # rounds to 0.76
        "do_predict": 1,
        "di_value": 0.118, # rounds to 0.12
        "volume_anomaly": True,
        "volatility_anomaly": False,
        "high_atr": True
    }

    hash1 = cache.get_hash("DOGE/USDT", "5m", ind1)
    hash2 = cache.get_hash("DOGE/USDT", "5m", ind2)

    # Hash should be identical due to discretization rounding
    assert hash1 == hash2

    # Verify per-pair isolation
    hash_other_pair = cache.get_hash("SOL/USDT", "5m", ind1)
    assert hash1 != hash_other_pair

    # Test lookup and update
    verdict = {"veto": False, "confidence": 0.85, "reason": "strong trend"}
    cache.update("DOGE/USDT", hash1, verdict)

    # Hit for DOGE
    assert cache.lookup("DOGE/USDT", hash1) == verdict

    # Miss for SOL/USDT with same hash
    assert cache.lookup("SOL/USDT", hash1) is None

def test_decision_logger(tmp_path):
    db_file = os.path.join(tmp_path, "test_trades.sqlite")
    logger = DecisionLogger(db_file)

    decision = {
        "trade_id": None,
        "pair": "ADA/USDT",
        "timeframe": "5m",
        "tier": 2,
        "model_confidence": 0.71,
        "di_ok": 1,
        "llm_invoked": 1,
        "llm_provider": "groq",
        "llm_veto": 0,
        "llm_confidence": 0.9,
        "llm_reason": "news check passed",
        "risk_checks_passed": 1,
        "block_reason": None,
        "outcome": "approved"
    }

    logger.log_decision(decision)

    # Read back to check persistence
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT pair, tier, model_confidence, llm_provider, outcome, trade_id FROM ai_decisions")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "ADA/USDT"
    assert row[1] == 2
    assert row[2] == 0.71
    assert row[3] == "groq"
    assert row[4] == "approved"
    assert row[5] is None

    # Test link trade ID
    logger.link_trade_id("ADA/USDT", 42)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT trade_id FROM ai_decisions WHERE id = 1")
    row = cursor.fetchone()
    conn.close()

    assert row[0] == 42

@patch("user_data.ai_layer.llm_client.Groq")
@patch("user_data.ai_layer.llm_client.genai")
def test_llm_client_fallback(mock_genai, mock_groq_class):
    # Setup mocks
    mock_groq_instance = MagicMock()
    mock_groq_class.return_value = mock_groq_instance

    # Mock Groq choice completion
    mock_choice = MagicMock()
    mock_choice.message.content = '{"veto": false, "confidence": 0.95, "reason": "no issues found"}'
    mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]

    # Configure mock environment
    with patch.dict(os.environ, {"GROQ_API_KEY": "test_groq_key", "GEMINI_API_KEY": "test_gemini_key"}):
        client = LLMClient({})

        # 1. Test standard Groq success path
        res = client.query_model("TICK:DOGE", "System prompt", "DOGE/USDT")
        assert res["provider"] == "groq"
        assert res["veto"] is False
        assert res["confidence"] == 0.95
        assert res["reason"] == "no issues found"

        # 2. Test Groq failure triggers Gemini fallback
        mock_groq_instance.chat.completions.create.side_effect = Exception("Rate limited")

        # Mock Gemini success using the new SDK Client structure
        mock_gemini_client = MagicMock()
        client.gemini_client = mock_gemini_client
        mock_gemini_res = MagicMock()
        mock_gemini_res.text = '{"veto": true, "confidence": 0.88, "reason": "high anomaly"}'
        mock_gemini_client.models.generate_content.return_value = mock_gemini_res

        res_fallback = client.query_model("TICK:DOGE", "System prompt", "DOGE/USDT")
        assert res_fallback["provider"] == "gemini"
        assert res_fallback["veto"] is True
        assert res_fallback["confidence"] == 0.88
        assert res_fallback["reason"] == "high anomaly"

        # 3. Test both fail triggers fail-safe veto=True
        mock_gemini_client.models.generate_content.side_effect = Exception("Gemini down")
        res_both_fail = client.query_model("TICK:DOGE", "System prompt", "DOGE/USDT")
        assert res_both_fail["provider"] == "none"
        assert res_both_fail["veto"] is True
        assert res_both_fail["reason"] == "llm_provider_failure"

def test_reporting_main(tmp_path):
    db_file = os.path.join(tmp_path, "test_trades_report.sqlite")
    logger = DecisionLogger(db_file)

    decision = {
        "trade_id": None,
        "pair": "ADA/USDT",
        "timeframe": "5m",
        "tier": 2,
        "model_confidence": 0.71,
        "di_ok": 1,
        "llm_invoked": 1,
        "llm_provider": "groq",
        "llm_veto": 0,
        "llm_confidence": 0.9,
        "llm_reason": "news check passed",
        "risk_checks_passed": 1,
        "block_reason": None,
        "outcome": "approved"
    }
    logger.log_decision(decision)

    # We must mock trades table because it might not exist in the DB yet
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY,
            is_open INTEGER,
            close_profit_pct REAL,
            close_date DATETIME
        )
    """)
    cursor.execute("INSERT INTO trades (id, is_open, close_profit_pct, close_date) VALUES (42, 0, 0.03, '2026-08-08')")
    conn.commit()
    conn.close()

    # Link trade
    logger.link_trade_id("ADA/USDT", 42)

    # Call main with args
    with patch("sys.argv", ["reporting.py", "--db-path", db_file, "--output", "console"]):
        reporting_main()

    with patch("sys.argv", ["reporting.py", "--db-path", db_file, "--output", "csv"]):
        reporting_main()
        assert os.path.exists("ai_decisions_report.csv")
        # Cleanup CSV
        os.remove("ai_decisions_report.csv")

def test_reporting_timeframe_and_outcomes(tmp_path):
    from datetime import datetime, timedelta
    import pandas as pd
    db_file = os.path.join(tmp_path, "test_timeframes.sqlite")
    logger = DecisionLogger(db_file)

    # 1. Blocked decision
    logger.log_decision({
        "trade_id": None,
        "pair": "XRP/USDT",
        "timeframe": "5m",
        "tier": 1,
        "model_confidence": 0.85,
        "di_ok": 1,
        "llm_invoked": 0,
        "llm_provider": "none",
        "llm_veto": 0,
        "risk_checks_passed": 0,
        "block_reason": "Max open trades reached",
        "outcome": "blocked"
    })

    # 2. Vetoed decision
    logger.log_decision({
        "trade_id": None,
        "pair": "SOL/USDT",
        "timeframe": "5m",
        "tier": 3,
        "model_confidence": 0.90,
        "di_ok": 1,
        "llm_invoked": 1,
        "llm_provider": "gemini",
        "llm_veto": 1,
        "llm_reason": "RSI too high",
        "risk_checks_passed": 1,
        "block_reason": None,
        "outcome": "vetoed"
    })

    # Let's check timestamp update for the vetoed decision to simulate an older timestamp (e.g., 10 days ago)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    ten_days_ago = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE ai_decisions SET timestamp = ? WHERE pair = 'SOL/USDT'", (ten_days_ago,))
    conn.commit()
    conn.close()

    # Call main with --last 7days (should filter out SOL/USDT)
    with patch("sys.argv", ["reporting.py", "--db-path", db_file, "--last", "7days", "--output", "csv"]):
        reporting_main()
        assert os.path.exists("ai_decisions_report.csv")
        df = pd.read_csv("ai_decisions_report.csv")
        assert len(df) == 1
        assert df.iloc[0]["Pair"] == "XRP/USDT"
        assert "Blocked (Max open trades reached)" in df.iloc[0]["Outcome"]
        os.remove("ai_decisions_report.csv")

    # Call main with --last 14days (should include SOL/USDT)
    with patch("sys.argv", ["reporting.py", "--db-path", db_file, "--last", "14days", "--output", "csv"]):
        reporting_main()
        assert os.path.exists("ai_decisions_report.csv")
        df = pd.read_csv("ai_decisions_report.csv")
        assert len(df) == 2
        # Order is by timestamp DESC, so XRP/USDT (recent) first, then SOL/USDT (older)
        assert df.iloc[0]["Pair"] == "XRP/USDT"
        assert df.iloc[1]["Pair"] == "SOL/USDT"
        assert "Vetoed (RSI too high)" in df.iloc[1]["Outcome"]
        os.remove("ai_decisions_report.csv")
