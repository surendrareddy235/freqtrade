"""
Tests for user_data/ai_layer/risk_manager.py
"""
import pytest
import pandas as pd
import numpy as np
from user_data.ai_layer.risk_manager import RiskManager

def test_risk_manager_init():
    config = {
        "max_open_trades": 3,
        "risk_manager": {
            "max_daily_trades": 5,
            "daily_drawdown_limit": 0.05,
            "position_size_percent": 2.0,
            "correlation_threshold": 0.85,
            "min_balance_buffer": 1.5
        }
    }
    rm = RiskManager(config)
    assert rm.max_daily_trades == 5
    assert rm.daily_drawdown_limit == 0.05
    assert rm.position_size_percent == 0.02  # converted from 2.0%
    assert rm.correlation_threshold == 0.85
    assert rm.min_balance_buffer == 1.5


def test_risk_manager_position_size_one_is_one_percent():
    config = {"risk_manager": {"position_size_percent": 1.0}}
    rm = RiskManager(config)

    assert rm.position_size_percent == 0.01

def test_risk_manager_calculate_correlation():
    config = {}
    rm = RiskManager(config)

    # Highly correlated series
    candidate = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    open_p = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    corr = rm.calculate_correlation(candidate, open_p)
    assert corr > 0.99

    # Uncorrelated series
    np.random.seed(42)
    candidate_rand = pd.Series(np.random.randn(20))
    open_rand = pd.Series(np.random.randn(20))
    corr_rand = rm.calculate_correlation(candidate_rand, open_rand)
    assert abs(corr_rand) < 0.5

def test_risk_manager_check_all_rules_sequential():
    config = {
        "max_open_trades": 3,
        "risk_manager": {
            "max_daily_trades": 5,
            "daily_drawdown_limit": 0.05,
            "position_size_percent": 10.0,  # 10%
            "correlation_threshold": 0.85,
            "min_balance_buffer": 1.5
        }
    }
    rm = RiskManager(config)

    # 1. Max open trades
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=3,
        daily_trades=0,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=10.0,
        available_balance=1000.0,
        min_notional=10.0,
        is_correlated=False,
        is_dry_run=True
    )
    assert not passed
    assert reason == "max_open_trades_exceeded"
    assert breakdown["max_open_trades"] is False
    assert breakdown["max_daily_trades"] is True
    assert breakdown["daily_drawdown"] is True

    # 2. Max trades per day
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=1,
        daily_trades=5,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=10.0,
        available_balance=1000.0,
        min_notional=10.0,
        is_correlated=False,
        is_dry_run=True
    )
    assert not passed
    assert reason == "max_trades_per_day_exceeded"
    assert breakdown["max_open_trades"] is True
    assert breakdown["max_daily_trades"] is False

    # 3. Daily drawdown
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=1,
        daily_trades=2,
        daily_drawdown_pct=0.06,  # 6% drawdown
        max_open_trades=3,
        proposed_stake=10.0,
        available_balance=1000.0,
        min_notional=10.0,
        is_correlated=False,
        is_dry_run=True
    )
    assert not passed
    assert reason == "daily_drawdown_limit_exceeded"
    assert breakdown["max_open_trades"] is True
    assert breakdown["max_daily_trades"] is True
    assert breakdown["daily_drawdown"] is False

    # 4. Position sizing (proposed_stake > available_balance * position_size_percent)
    # limit is 1000.0 * 0.10 = 100.0 USDT
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=1,
        daily_trades=2,
        daily_drawdown_pct=0.02,
        max_open_trades=3,
        proposed_stake=150.0,  # exceeds 100.0
        available_balance=1000.0,
        min_notional=10.0,
        is_correlated=False,
        is_dry_run=True
    )
    assert not passed
    assert reason == "position_size_exceeded"
    assert breakdown["position_sizing"] is False

    # 5. Correlation check
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=1,
        daily_trades=2,
        daily_drawdown_pct=0.02,
        max_open_trades=3,
        proposed_stake=50.0,
        available_balance=1000.0,
        min_notional=10.0,
        is_correlated=True,
        is_dry_run=True
    )
    assert not passed
    assert reason == "correlation_veto"
    assert breakdown["correlation"] is False

    # 6. LLM veto hook (stub/default verdict of False)
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=1,
        daily_trades=2,
        daily_drawdown_pct=0.02,
        max_open_trades=3,
        proposed_stake=50.0,
        available_balance=1000.0,
        min_notional=10.0,
        is_correlated=False,
        is_dry_run=True,
        llm_veto=True
    )
    assert not passed
    assert reason == "llm_veto"
    assert breakdown["llm_veto"] is False

    # 7. Minimum-viable-balance check
    # Minimum balance limit = min_notional * min_balance_buffer = 10.0 * 1.5 = 15.0 USDT
    # Bypassed in dry run
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=1,
        daily_trades=2,
        daily_drawdown_pct=0.02,
        max_open_trades=3,
        proposed_stake=0.5,  # Within the 10% of 10.0 (1.0 limit) position size cap
        available_balance=10.0,  # Below 15.0 USDT but is dry run
        min_notional=10.0,
        is_correlated=False,
        is_dry_run=True
    )
    assert passed
    assert reason == "passed"
    assert breakdown["min_balance"] is True

    # Triggered in live mode (is_dry_run=False)
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=1,
        daily_trades=2,
        daily_drawdown_pct=0.02,
        max_open_trades=3,
        proposed_stake=0.5,  # Within the 10% of 10.0 (1.0 limit) position size cap
        available_balance=10.0,  # Below 15.0 USDT
        min_notional=10.0,
        is_correlated=False,
        is_dry_run=False
    )
    assert not passed
    assert reason == "insufficient_balance"
    assert breakdown["min_balance"] is False


def test_risk_manager_position_sizing_leverage():
    # position size percent = 10% of 1000 USDT = 100 USDT allowed margin
    config = {
        "stoploss": -0.01,
        "risk_manager": {
            "position_size_percent": 10.0,
        }
    }
    rm = RiskManager(config)

    # 1. Without leverage: proposed_stake 150 > 100 allowed (fails)
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=0,
        daily_trades=0,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=150.0,
        available_balance=1000.0,
        min_notional=5.0,
        is_correlated=False,
        leverage=1.0,
        open_rate=50000.0
    )
    assert not passed
    assert reason == "position_size_exceeded"
    assert breakdown["position_sizing"] is False

    # 2. With leverage 3x: proposed_stake 150 -> margin 50 <= 100 allowed (passes)
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=0,
        daily_trades=0,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=150.0,
        available_balance=1000.0,
        min_notional=5.0,
        is_correlated=False,
        leverage=3.0,
        open_rate=50000.0
    )
    assert passed
    assert reason == "passed"
    assert breakdown["position_sizing"] is True


def test_risk_manager_liquidation_distance_check():
    # Stoploss = -1% (0.01). 2x stoploss is 2% (0.02)
    config = {
        "stoploss": -0.01,
        "risk_manager": {
            "position_size_percent": 10.0,  # 10%
        }
    }
    rm = RiskManager(config)

    # 1. Leverage = 3, MMR = 0.05. Safe distance (approx 29.8% from entry) -> passes
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=0,
        daily_trades=0,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=150.0,
        available_balance=1000.0,
        min_notional=5.0,
        is_correlated=False,
        leverage=3.0,
        open_rate=50000.0,
        mm_ratio=0.05
    )
    assert passed
    assert reason == "passed"
    assert breakdown["liquidation_distance"] is True

    # 2. Extremely high leverage (e.g. 100x), liquidation distance is too close (< 2% stoploss safety) -> blocked
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=0,
        daily_trades=0,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=150.0,
        available_balance=1000.0,
        min_notional=5.0,
        is_correlated=False,
        leverage=100.0,
        open_rate=50000.0,
        mm_ratio=0.05
    )
    assert not passed
    assert reason == "liquidation_risk"
    assert breakdown["liquidation_distance"] is False


def test_risk_manager_futures_min_notional_margin():
    config = {
        "risk_manager": {
            "min_balance_buffer": 1.5,
            "position_size_percent": 100.0,  # Allow 100% position size for min balance check
        }
    }
    rm = RiskManager(config)

    # In live mode (is_dry_run=False), futures minimum viable margin check:
    # 5 USDT min_notional * 1.5 buffer = 7.5 USDT required min balance

    # 1. Balance = 8.0 USDT >= 7.5 USDT -> passes
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=0,
        daily_trades=0,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=10.0,
        available_balance=8.0,
        min_notional=5.0,
        is_correlated=False,
        is_dry_run=False,
        leverage=3.0,
        open_rate=50000.0
    )
    assert passed
    assert reason == "passed"
    assert breakdown["min_balance"] is True

    # 2. Balance = 7.0 USDT < 7.5 USDT -> blocked
    passed, reason, breakdown = rm.check_all_rules(
        pair="BTC/USDT",
        current_open_trades=0,
        daily_trades=0,
        daily_drawdown_pct=0.0,
        max_open_trades=3,
        proposed_stake=10.0,
        available_balance=7.0,
        min_notional=5.0,
        is_correlated=False,
        is_dry_run=False,
        leverage=3.0,
        open_rate=50000.0
    )
    assert not passed
    assert reason == "insufficient_balance"
    assert breakdown["min_balance"] is False
