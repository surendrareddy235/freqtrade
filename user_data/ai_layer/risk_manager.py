"""
risk_manager.py

Houses the Risk Manager logic. Pure functions representing deterministic risk checks (no side-effects beyond logging).
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Validates trade candidates against strict, non-overridable risk rules.
    """
    def __init__(self, config: dict):
        self.config = config.get("risk_manager", {}) if config else {}

        # Max open trades is read from Freqtrade's top-level config, not from a separate risk block.
        # But we can store a default in case it is missing.
        self.max_open_trades = config.get("max_open_trades", 1) if config else 1

        # 2. Max trades per day
        self.max_trades_per_day = self.config.get("max_trades_per_day", self.config.get("daily_trade_cap", 10))

        # 3. Daily drawdown limit (ratio, e.g. 0.05 for 5%)
        self.daily_drawdown_limit = self.config.get("daily_drawdown_limit", self.config.get("drawdown_limit_pct", 5.0))
        if self.daily_drawdown_limit > 1.0:
            self.daily_drawdown_limit /= 100.0

        # 4. Position size limit. Values >= 1 are percentages (1.0 means 1%).
        # Ratios below 1 are accepted directly (0.01 also means 1%).
        self.position_size_percent = self.config.get("position_size_percent", self.config.get("position_sizing_pct", 1.0))
        if self.position_size_percent >= 1.0:
            self.position_size_percent /= 100.0

        # 5. Correlation threshold
        self.correlation_threshold = self.config.get("correlation_threshold", 0.85)

        # 7. Minimum balance buffer
        self.min_balance_buffer = self.config.get("min_balance_buffer", 1.5)

    def calculate_correlation(self, candidate_returns: pd.Series, open_returns: pd.Series) -> float:
        """
        Calculates a real deterministic Pearson correlation on close returns.
        """
        try:
            merged = pd.DataFrame({"candidate": candidate_returns, "open": open_returns}).dropna()
            if len(merged) >= 10:
                return float(merged["candidate"].corr(merged["open"]))
        except Exception as e:
            logger.error(f"Error calculating Pearson correlation: {e}")
        return 0.0

    def check_all_rules(self, pair: str, current_open_trades: int, daily_trades: int,
                        daily_drawdown_pct: float, max_open_trades: int,
                        proposed_stake: float, available_balance: float, min_notional: float,
                        is_correlated: bool, is_dry_run: bool = True,
                        llm_veto: bool = False) -> tuple[bool, str]:
        """
        Runs all deterministic checks in the exact specified order:
        1. Max open trades
        2. Max trades per day
        3. Daily drawdown kill-switch (5%)
        4. Per-trade position sizing
        5. Correlation check
        6. LLM veto check
        7. Minimum-viable-balance check (Bypassed in dry-run mode)

        Returns tuple (passed: bool, reason: str).
        """
        # 1. Max open trades
        if current_open_trades >= max_open_trades:
            msg = "max_open_trades_exceeded"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (open={current_open_trades}, max={max_open_trades})")
            return False, msg

        # 2. Max trades per day
        if daily_trades >= self.max_trades_per_day:
            msg = "max_trades_per_day_exceeded"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (trades_today={daily_trades}, cap={self.max_trades_per_day})")
            return False, msg

        # 3. Daily drawdown kill-switch
        if daily_drawdown_pct >= self.daily_drawdown_limit:
            msg = "daily_drawdown_limit_exceeded"
            logger.warning(f"Trade blocked for {pair}. Reason: {msg} (drawdown={daily_drawdown_pct:.2%}, limit={self.daily_drawdown_limit:.2%})")
            return False, msg

        # 4. Per-trade position sizing check (hard cap)
        max_allowed_stake = available_balance * self.position_size_percent
        if proposed_stake > max_allowed_stake:
            msg = "position_size_exceeded"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (proposed_stake={proposed_stake:.2f}, limit={max_allowed_stake:.2f})")
            return False, msg

        # 5. Correlation check
        if is_correlated:
            msg = "correlation_veto"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (pair is highly correlated with an already-open trade)")
            return False, msg

        # 6. LLM veto check (stub/default verdict of False)
        if llm_veto:
            msg = "llm_veto"
            logger.info(f"Trade blocked for {pair}. Reason: {msg}")
            return False, msg

        # 7. Minimum-viable-balance check (Bypassed in dry-run mode)
        if not is_dry_run:
            required_min = min_notional * self.min_balance_buffer
            if available_balance < required_min:
                msg = "insufficient_balance"
                logger.warning(f"Trade blocked for {pair}. Reason: {msg} (balance={available_balance:.2f}, required={required_min:.2f})")
                return False, msg

        logger.info(f"All risk checks passed for {pair}")
        return True, "passed"
