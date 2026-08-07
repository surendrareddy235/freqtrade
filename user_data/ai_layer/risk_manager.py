"""
risk_manager.py

Houses the Risk Manager logic. Pure functions representing deterministic risk checks (no side-effects beyond logging).
"""
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Validates trade candidates against strict, non-overridable risk rules.
    """
    def __init__(self, config: dict):
        self.config = config.get("risk_manager", {})
        self.drawdown_limit_pct = self.config.get("drawdown_limit_pct", 5.0) / 100.0
        self.daily_trade_cap = self.config.get("daily_trade_cap", 10)
        self.position_sizing_pct = self.config.get("position_sizing_pct", 1.0) / 100.0
        self.correlation_threshold = self.config.get("correlation_threshold", 0.85)
        self.min_balance_buffer = self.config.get("min_balance_buffer", 1.5)

    def check_all_rules(self, pair: str, current_open_trades: int, daily_trades: int,
                        daily_drawdown_pct: float, max_open_trades: int,
                        available_balance: float, min_notional: float,
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
            msg = f"max_open_trades_exceeded (open={current_open_trades}, max={max_open_trades})"
            logger.info(msg)
            return False, msg

        # 2. Max trades per day
        if daily_trades >= self.daily_trade_cap:
            msg = f"daily_trade_cap_exceeded (trades_today={daily_trades}, cap={self.daily_trade_cap})"
            logger.info(msg)
            return False, msg

        # 3. Daily drawdown kill-switch
        if daily_drawdown_pct >= self.drawdown_limit_pct:
            msg = f"daily_drawdown_kill_switch_active (drawdown={daily_drawdown_pct:.2%}, limit={self.drawdown_limit_pct:.2%})"
            logger.warning(msg)
            return False, msg

        # 4. Per-trade position sizing check (Informational/Logical verification)
        proposed_size = available_balance * self.position_sizing_pct
        if proposed_size <= 0:
            msg = f"invalid_position_size (proposed_size={proposed_size}, balance={available_balance})"
            logger.info(msg)
            return False, msg

        # 5. Correlation check
        if is_correlated:
            msg = f"correlation_veto (pair {pair} is highly correlated with an already-open trade)"
            logger.info(msg)
            return False, msg

        # 6. LLM veto check (placeholder - always passes in Phase 3)
        if llm_veto:
            msg = "llm_veto_triggered"
            logger.info(msg)
            return False, msg

        # 7. Minimum-viable-balance check (Bypassed in dry-run mode)
        if not is_dry_run:
            required_min = min_notional * self.min_balance_buffer
            if available_balance < required_min:
                msg = f"insufficient_balance (balance={available_balance}, required={required_min})"
                logger.warning(msg)
                return False, msg

        logger.info(f"All risk checks passed for {pair}")
        return True, "passed"
