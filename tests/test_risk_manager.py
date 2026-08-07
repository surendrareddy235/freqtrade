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
            "max_trades_per_day": 5,
            "daily_drawdown_limit": 0.05,
            "position_size_percent": 2.0,
            "correlation_threshold": 0.85,
            "min_balance_buffer": 1.5
        }
    }
    rm = RiskManager(config)
    assert rm.max_trades_per_day == 5
    assert rm.daily_drawdown_limit == 0.05
    assert rm.position_size_percent == 0.02  # converted from 2.0 %
    assert rm.correlation_threshold == 0.85
    assert rm.min_balance_buffer == 1.5

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
            "max_trades_per_day": 5,
            "daily_drawdown_limit": 0.05,
            "position_size_percent": 10.0,  # 10%
            "correlation_threshold": 0.85,
            "min_balance_buffer": 1.5
        }
    }
    rm = RiskManager(config)

    # 1. Max open trades
    passed, reason = rm.check_all_rules(
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

    # 2. Max trades per day
    passed, reason = rm.check_all_rules(
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

    # 3. Daily drawdown
    passed, reason = rm.check_all_rules(
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

    # 4. Position sizing (proposed_stake > available_balance * position_size_percent)
    # limit is 1000.0 * 0.10 = 100.0 USDT
    passed, reason = rm.check_all_rules(
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

    # 5. Correlation check
    passed, reason = rm.check_all_rules(
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

    # 6. LLM veto hook (stub/default verdict of False)
    passed, reason = rm.check_all_rules(
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

    # 7. Minimum-viable-balance check
    # Minimum balance limit = min_notional * min_balance_buffer = 10.0 * 1.5 = 15.0 USDT
    # Bypassed in dry run
    passed, reason = rm.check_all_rules(
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

    # Triggered in live mode (is_dry_run=False)
    passed, reason = rm.check_all_rules(
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
