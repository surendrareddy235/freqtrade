# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import sys
import os
sys.path.append(os.getcwd())

import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union
import talib.abstract as ta

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

# Import our custom RiskManager and DecisionLogger
from user_data.ai_layer.risk_manager import RiskManager
from user_data.ai_layer.decision_logger import DecisionLogger

logger = logging.getLogger(__name__)

class ScalpStrategy(IStrategy):
    """
    ScalpStrategy - Custom IStrategy class for AI-Driven Crypto Scalping Bot.
    This strategy represents the base setup for Phase 3.
    """

    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short? Spot trading only for v1, so False.
    can_short: bool = False

    # Minimal ROI targets and Stoploss per §B.4a (Fixed 1:3 ratio, no decaying targets)
    minimal_roi = {
        "0": 0.03  # Flat 3% profit target
    }
    stoploss = -0.01  # Flat -1% stop loss

    # Trailing stop is disabled per §B.4a (conflicts with a strict fixed ratio)
    trailing_stop = False

    # Process only on candle close to prevent excessive compute
    process_only_new_candles = True

    # Startup candles needed for technical indicators
    startup_candle_count: int = 100

    # Risk state cache refreshed in bot_loop_start() and used in confirm_trade_entry()
    daily_trades_count = 0
    daily_drawdown_pct = 0.0
    current_open_trades_count = 0

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.risk_manager = RiskManager(config)
        self.decision_logger = DecisionLogger(config.get("db_url", "tradesv3.dryrun.sqlite"))
        self.close_returns_cache = {}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate base indicators that feed into FreqAI feature pipeline.
        In Phase 3, this starts the FreqAI pipeline. Do not put feature engineering here
        to avoid look-ahead bias as per §B.3.
        """
        self.freqai_info = self.config["freqai"]
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        """
        FreqAI custom feature engineering pipeline.
        Must prefix engineered feature columns with '%' as per §B.3.
        We have exactly 29 base features mapped across Trend, Momentum, Volatility, Volume, and Price action.
        """
        epsilon = 1e-8

        # --- 1. Trend (Total 8 features) ---
        # EMA 9, 21, 50
        dataframe["%ema-9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["%ema-21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["%ema-50"] = ta.EMA(dataframe, timeperiod=50)

        # EMA slopes
        dataframe["%ema-slope-9"] = dataframe["%ema-9"].diff()
        dataframe["%ema-slope-21"] = dataframe["%ema-21"].diff()
        dataframe["%ema-slope-50"] = dataframe["%ema-50"].diff()

        # Custom EMA using config period
        dataframe[f"%ema-{period}"] = ta.EMA(dataframe, timeperiod=period)
        dataframe[f"%ema-slope-{period}"] = dataframe[f"%ema-{period}"].diff()

        # VWAP relative difference (expose as normalized feature: (close - VWAP) / VWAP)
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        pv = typical_price * dataframe["volume"]
        cum_pv = pv.cumsum()
        cum_vol = dataframe["volume"].cumsum()
        vwap = cum_pv / (cum_vol + epsilon)
        dataframe["%vwap-diff"] = (dataframe["close"] - vwap) / (vwap + epsilon)

        # --- 2. Momentum (Total 5 features) ---
        dataframe["%rsi-14"] = ta.RSI(dataframe, timeperiod=14)

        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14)
        dataframe["%stoch-rsi-k"] = stoch_rsi["fastk"]
        dataframe["%stoch-rsi-d"] = stoch_rsi["fastd"]

        dataframe["%cci-14"] = ta.CCI(dataframe, timeperiod=14)
        dataframe["%roc-14"] = ta.ROC(dataframe, timeperiod=14)

        # --- 3. Volatility (Total 5 features) ---
        dataframe["%atr-14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["%natr-14"] = ta.NATR(dataframe, timeperiod=14)

        bb = ta.BBANDS(dataframe, timeperiod=14)
        dataframe["%bb-upper"] = bb["upperband"]
        dataframe["%bb-middle"] = bb["middleband"]
        dataframe["%bb-lower"] = bb["lowerband"]

        # %B and BB Width
        dataframe["%bb-pct-b"] = (dataframe["close"] - dataframe["%bb-lower"]) / (dataframe["%bb-upper"] - dataframe["%bb-lower"] + epsilon)
        dataframe["%bb-width"] = (dataframe["%bb-upper"] - dataframe["%bb-lower"]) / (dataframe["%bb-middle"] + epsilon)

        # --- 4. Volume (Total 4 features) ---
        dataframe["%volume"] = dataframe["volume"]
        dataframe["%volume-ema-14"] = ta.EMA(dataframe, price="volume", timeperiod=14)
        dataframe["%obv"] = ta.OBV(dataframe)

        # Relative Volume (ratio of volume to its rolling 14-period mean)
        rolling_vol_mean = dataframe["volume"].rolling(14).mean()
        dataframe["%relative-volume"] = dataframe["volume"] / (rolling_vol_mean + epsilon)

        # --- 5. Price Action (Total 7 features) ---
        candle_range = dataframe["high"] - dataframe["low"]
        dataframe["%candle-body-pct"] = np.abs(dataframe["close"] - dataframe["open"]) / (candle_range + epsilon)

        # Wick percentages
        upper_wick = np.where(dataframe["close"] > dataframe["open"], dataframe["high"] - dataframe["close"], dataframe["high"] - dataframe["open"])
        lower_wick = np.where(dataframe["close"] > dataframe["open"], dataframe["open"] - dataframe["low"], dataframe["close"] - dataframe["low"])
        dataframe["%upper-wick-pct"] = upper_wick / (candle_range + epsilon)
        dataframe["%lower-wick-pct"] = lower_wick / (candle_range + epsilon)

        # High-low range percentage
        dataframe["%high-low-range-pct"] = candle_range / (dataframe["close"] + epsilon)

        # Returns over 1, 3, 5 candles
        dataframe["%returns-1"] = dataframe["close"].pct_change(1)
        dataframe["%returns-3"] = dataframe["close"].pct_change(3)
        dataframe["%returns-5"] = dataframe["close"].pct_change(5)

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        FreqAI basic feature engineering pipeline. Not used for features to keep count under 35.
        """
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        FreqAI standard feature engineering pipeline.
        Exotic non-timeframe-expanded features can go here. We use it to populate time of day features
        or day of week if needed, but we keep features minimal to avoid exceeding the 35 feature limit.
        We will return dataframe as is.
        """
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        FreqAI target labelling. Must prefix target columns with '&' as per §B.3.
        Predicts whether the price moves >=0.7% within the next 8 candles.
        This is implemented as a binary classification target (1 if True, 0 otherwise).
        """
        # Calculate max close/high return over the next 8 candles relative to current close
        future_max_high = dataframe["high"].shift(-8).rolling(8).max()
        future_return = (future_max_high - dataframe["close"]) / dataframe["close"]

        dataframe["&target"] = np.where(future_return >= 0.007, 1, 0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Phase 2 entry logic: use model predictions for entry.
        Buy if the model predicts the target is 1 (upward move >= 0.7% expected)
        and the data kitchen do_predict flag is true (not outlier, no NaNs).
        """
        dataframe.loc[
            (dataframe["&target"] == 1) & (dataframe["do_predict"] == 1),
            ["enter_long", "enter_tag"]
        ] = (1, "long_scalp")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Phase 2 exit logic: exits are ROI/stoploss driven natively.
        No custom exit signal.
        """
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = ""
        return dataframe

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """
        Refreshes cached risk state at each loop step before entering 'confirm_trade_entry' callbacks.
        Calculates daily trade counts, open trade counts, and daily drawdown percentage relative to start-of-day equity.
        Precomputes close returns for whitelisted pairs and open positions to avoid heavy computations in confirm_trade_entry.
        """
        try:
            # Query all trades (both open and closed)
            all_trades = Trade.get_trades_proxy()

            # 1. Active/Open trade count
            self.current_open_trades_count = len([t for t in all_trades if t.is_open])

            # 2. Trades taken or closed today (UTC timezone)
            today_start = current_time.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            daily_trades = []
            for t in all_trades:
                if t.open_date_utc >= today_start:
                    daily_trades.append(t)
                elif t.close_date_utc is not None and t.close_date_utc >= today_start:
                    daily_trades.append(t)
            self.daily_trades_count = len(daily_trades)

            # 3. Drawdown calculator relative to start-of-day equity
            # Sum of closed profit/loss today
            daily_closed_profit = sum([t.close_profit_abs for t in daily_trades if not t.is_open and t.close_profit_abs is not None])

            # Sum of unrealized profit/loss today for trades opened today
            daily_unrealized_profit = 0.0

            # Total unrealized profit of all currently open trades (used for current equity calculation)
            total_unrealized_profit = 0.0

            for t in all_trades:
                if t.is_open:
                    df_open, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                    if df_open is not None and not df_open.empty:
                        current_rate = df_open.iloc[-1]["close"]
                        unrealized_profit_abs = t.calc_profit_ratio(current_rate) * t.stake_amount
                        total_unrealized_profit += unrealized_profit_abs
                        if t.open_date_utc >= today_start:
                            daily_unrealized_profit += unrealized_profit_abs

            total_profit_today = daily_closed_profit + daily_unrealized_profit

            # Get free stake balance
            free_balance = None
            if hasattr(self, "wallets") and self.wallets is not None:
                try:
                    free_balance = self.wallets.get_free(self.config.get("stake_currency", "USDT"))
                except Exception:
                    pass
            if free_balance is None:
                free_balance = self.config.get("dry_run_wallet", 1000.0)

            # Calculate total open trades cost (initial stake amount)
            total_open_trades_cost = sum([t.stake_amount for t in all_trades if t.is_open])

            # Current portfolio equity
            current_equity = free_balance + total_open_trades_cost + total_unrealized_profit

            # Start of day equity
            start_of_day_equity = current_equity - total_profit_today

            # Calculate daily drawdown percentage relative to start-of-day equity
            if start_of_day_equity > 0:
                self.daily_drawdown_pct = max(0.0, -total_profit_today / start_of_day_equity)
            else:
                self.daily_drawdown_pct = 0.0

            logger.info(
                f"[RiskState] Current Open Trades: {self.current_open_trades_count}, "
                f"Daily Trades Count: {self.daily_trades_count}, "
                f"Daily Drawdown: {self.daily_drawdown_pct:.2%}, "
                f"Start-of-Day Equity: {start_of_day_equity:.2f}, "
                f"Current Equity: {current_equity:.2f}"
            )

            # 4. Precompute and cache the last 30 candles of close returns
            self.close_returns_cache = {}
            open_pairs = [t.pair for t in all_trades if t.is_open]
            whitelist = self.dp.current_whitelist() if self.dp else []
            all_pairs = list(set(whitelist + open_pairs))

            for pair in all_pairs:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is not None and not df.empty:
                    close_returns = df["close"].pct_change()
                    self.close_returns_cache[pair] = close_returns.tail(30)

        except Exception as e:
            logger.error(f"Error in bot_loop_start risk state computation: {e}", exc_info=True)
            self.current_open_trades_count = 0
            self.daily_trades_count = 0
            self.daily_drawdown_pct = 0.0
            self.close_returns_cache = {}

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Runs the final veto check using RiskManager rules. Bypasses balance checks in dry-run mode.
        Performs exact rolling returns correlation check against currently open trades using cached data.
        """
        try:
            is_dry_run = self.config.get("dry_run", True)
            max_open_trades = self.config.get("max_open_trades", 1)

            # 1. Fetch candidate returns
            candidate_returns = getattr(self, "close_returns_cache", {}).get(pair)
            if candidate_returns is None:
                # Fallback to fetch on-demand if missing in cache
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if df is not None and not df.empty:
                    candidate_returns = df["close"].pct_change().tail(30)

            # 2. Calculate correlation check
            is_correlated = False
            all_trades = Trade.get_trades_proxy()
            open_trades = [t for t in all_trades if t.is_open]

            if open_trades and candidate_returns is not None:
                for ot in open_trades:
                    if ot.pair == pair:
                        continue
                    open_returns = getattr(self, "close_returns_cache", {}).get(ot.pair)
                    if open_returns is None:
                        df_open, _ = self.dp.get_analyzed_dataframe(ot.pair, self.timeframe)
                        if df_open is not None and not df_open.empty:
                            open_returns = df_open["close"].pct_change().tail(30)

                    if open_returns is not None:
                        correlation_value = self.risk_manager.calculate_correlation(candidate_returns, open_returns)
                        logger.info(f"[Risk Correlation] {pair} vs {ot.pair} correlation is {correlation_value:.2f}")
                        if correlation_value > self.risk_manager.correlation_threshold:
                            is_correlated = True
                            break

            # 3. Fetch minimum order notional limit for the exchange
            min_notional = 10.0  # Safe robust baseline default for USDT spot markets
            try:
                if hasattr(self.dp, "market") and self.dp.market(pair) is not None:
                    market_info = self.dp.market(pair)
                    min_notional = market_info.get("limits", {}).get("cost", {}).get("min", 10.0) or 10.0
            except Exception:
                pass

            # 4. Wallet available balance
            available_balance = None
            if hasattr(self, "wallets") and self.wallets is not None:
                try:
                    available_balance = self.wallets.get_free(self.config.get("stake_currency", "USDT"))
                except Exception:
                    pass
            if available_balance is None:
                available_balance = self.config.get("dry_run_wallet", 1000.0)

            # 5. Proposed stake amount
            proposed_stake = amount * rate

            # 6. Execute all RiskManager checks sequentially
            passed, reason = self.risk_manager.check_all_rules(
                pair=pair,
                current_open_trades=self.current_open_trades_count,
                daily_trades=self.daily_trades_count,
                daily_drawdown_pct=self.daily_drawdown_pct,
                max_open_trades=max_open_trades,
                proposed_stake=proposed_stake,
                available_balance=available_balance,
                min_notional=min_notional,
                is_correlated=is_correlated,
                is_dry_run=is_dry_run,
                llm_veto=False
            )

            # 7. Record structured decision in SQLite database
            decision_data = {
                "trade_id": None,
                "pair": pair,
                "timeframe": self.timeframe,
                "tier": 1,
                "model_confidence": 1.0,
                "di_ok": 1,
                "llm_invoked": 0,
                "llm_provider": "none",
                "llm_veto": 0,
                "llm_confidence": 0.0,
                "llm_reason": "none",
                "risk_checks_passed": 1 if passed else 0,
                "block_reason": reason if not passed else None,
                "outcome": "approved" if passed else "rejected"
            }
            try:
                self.decision_logger.log_decision(decision_data)
            except Exception as db_err:
                logger.error(f"Failed to log decision to database: {db_err}")

            if not passed:
                logger.warning(f"[RiskVeto] Trade blocked for {pair}. Reason: {reason}")
                return False

            logger.info(f"[RiskSuccess] Trade entry confirmed for {pair}")
            return True

        except Exception as e:
            logger.error(f"Error in confirm_trade_entry Risk Manager checks: {e}", exc_info=True)
            # Fail closed for maximum safety
            return False
