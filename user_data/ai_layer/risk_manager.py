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
        self.max_daily_trades = self.config.get("max_daily_trades", 10)

        # 3. Daily drawdown limit (ratio, e.g. 0.05 for 5%)
        self.daily_drawdown_limit = self.config.get("daily_drawdown_limit", 0.05)
        if self.daily_drawdown_limit > 1.0:
            self.daily_drawdown_limit /= 100.0

        # 4. Position size limit. Values >= 1 are percentages (1.0 means 1%).
        # Ratios below 1 are accepted directly (0.01 also means 1%).
        self.position_size_percent = self.config.get("position_size_percent", 1.0)
        if self.position_size_percent >= 1.0:
            self.position_size_percent /= 100.0

        # 5. Correlation threshold
        self.correlation_threshold = self.config.get("correlation_threshold", 0.85)

        # 7. Minimum balance buffer
        self.min_balance_buffer = self.config.get("min_balance_buffer", 1.5)

        # Stoploss configuration from top-level config (e.g. -0.01 for -1%)
        self.stoploss = abs(config.get("stoploss", -0.01)) if config else 0.01

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
                        llm_veto: bool = False,
                        leverage: float = 1.0, open_rate: float = 0.0,
                        mm_ratio: float = 0.05, maint_amt: float = 0.0) -> tuple[bool, str, dict[str, bool]]:
        """
        Runs all deterministic checks in the exact specified order:
        1. Max open trades
        2. Max daily trades
        3. Daily drawdown kill-switch (5%)
        4. Per-trade position sizing (Leverage-adjusted margin required vs. wallet balance percent)
        5. Correlation check
        6. LLM veto check
        7. Liquidation-distance check (Pre-trade safety gate before liquidation risk)
        8. Minimum-viable-balance check (Bypassed in dry-run mode, uses futures minimum notional)

        Returns tuple (passed: bool, reason: str, breakdown: dict[str, bool]).
        """
        # 1. Max open trades
        check_max_open = current_open_trades < max_open_trades

        # 2. Max daily trades
        check_max_daily = daily_trades < self.max_daily_trades

        # 3. Daily drawdown kill-switch
        check_daily_drawdown = daily_drawdown_pct < self.daily_drawdown_limit

        # 4. Position sizing check (leverage-adjusted margin vs wallet balance percent)
        proposed_margin = proposed_stake / leverage if leverage > 0 else proposed_stake
        max_allowed_margin = available_balance * self.position_size_percent
        check_position_sizing = proposed_margin <= max_allowed_margin

        # 5. Correlation check
        check_correlation = not is_correlated

        # 6. LLM veto check
        check_llm_veto = not llm_veto

        # 7. Liquidation-distance check (§B.6.8)
        check_liq_distance = True
        if open_rate > 0 and leverage > 0:
            numerator = open_rate * (1.0 - 1.0 / leverage)
            if proposed_stake > 0:
                numerator -= (maint_amt * open_rate) / proposed_stake
            liq_price = numerator / (1.0 - mm_ratio)

            if liq_price >= open_rate or liq_price <= 0:
                liq_distance = 0.0
            else:
                liq_distance = (open_rate - liq_price) / open_rate

            required_distance = 2.0 * self.stoploss
            if liq_distance < required_distance:
                check_liq_distance = False

        # 8. Minimum-viable-balance check (Bypassed in dry-run mode)
        check_min_balance = True
        if not is_dry_run:
            required_min = min_notional * self.min_balance_buffer
            if available_balance < required_min:
                check_min_balance = False

        breakdown = {
            "max_open_trades": bool(check_max_open),
            "max_daily_trades": bool(check_max_daily),
            "daily_drawdown": bool(check_daily_drawdown),
            "position_sizing": bool(check_position_sizing),
            "correlation": bool(check_correlation),
            "llm_veto": bool(check_llm_veto),
            "liquidation_distance": bool(check_liq_distance),
            "min_balance": bool(check_min_balance)
        }

        # Sequential blocking in exact order:
        if not check_max_open:
            msg = "max_open_trades_exceeded"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (open={current_open_trades}, max={max_open_trades})")
            return False, msg, breakdown

        if not check_max_daily:
            msg = "max_trades_per_day_exceeded"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (trades_today={daily_trades}, cap={self.max_daily_trades})")
            return False, msg, breakdown

        if not check_daily_drawdown:
            msg = "daily_drawdown_limit_exceeded"
            logger.warning(f"Trade blocked for {pair}. Reason: {msg} (drawdown={daily_drawdown_pct:.2%}, limit={self.daily_drawdown_limit:.2%})")
            return False, msg, breakdown

        if not check_position_sizing:
            msg = "position_size_exceeded"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (proposed_margin={proposed_margin:.2f}, limit={max_allowed_margin:.2f})")
            return False, msg, breakdown

        if not check_correlation:
            msg = "correlation_veto"
            logger.info(f"Trade blocked for {pair}. Reason: {msg} (pair is highly correlated with an already-open trade)")
            return False, msg, breakdown

        if not check_llm_veto:
            msg = "llm_veto"
            logger.info(f"Trade blocked for {pair}. Reason: {msg}")
            return False, msg, breakdown

        if not check_liq_distance:
            msg = "liquidation_risk"
            # Calculate liq_price again for log
            numerator = open_rate * (1.0 - 1.0 / leverage)
            if proposed_stake > 0:
                numerator -= (maint_amt * open_rate) / proposed_stake
            liq_price = numerator / (1.0 - mm_ratio)
            liq_distance = (open_rate - liq_price) / open_rate if (0 < liq_price < open_rate) else 0.0
            required_distance = 2.0 * self.stoploss
            logger.warning(f"Trade blocked for {pair}. Reason: {msg} (liq_price={liq_price:.4f}, liq_distance={liq_distance:.4%}, required={required_distance:.4%})")
            return False, msg, breakdown

        if not check_min_balance:
            msg = "insufficient_balance"
            required_min = min_notional * self.min_balance_buffer
            logger.warning(f"Trade blocked for {pair}. Reason: {msg} (balance={available_balance:.2f}, required={required_min:.2f})")
            return False, msg, breakdown

        logger.info(f"All risk checks passed for {pair}")
        return True, "passed", breakdown
