# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Union
import talib.abstract as ta

from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

class ScalpStrategy(IStrategy):
    """
    ScalpStrategy - Custom IStrategy class for AI-Driven Crypto Scalping Bot.
    This strategy represents the base setup for Phase 2.
    """

    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Can this strategy go short? Spot trading only for v1, so False.
    can_short: bool = False

    # Minimal ROI targets and Stoploss will be overridden by config_freqai.json
    # but defined here as fallbacks / temporary placeholders for Phase 2 as per §B.10.
    minimal_roi = {
        "0": 0.03  # 3% profit target
    }
    stoploss = -0.01  # -1% stop loss

    # Trailing stop is disabled for v1 as per §B.4a
    trailing_stop = False

    # Process only on candle close to prevent excessive compute
    process_only_new_candles = True

    # Startup candles needed for technical indicators
    startup_candle_count: int = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate base indicators that feed into FreqAI feature pipeline.
        In Phase 2, this starts the FreqAI pipeline. Do not put feature engineering here
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
