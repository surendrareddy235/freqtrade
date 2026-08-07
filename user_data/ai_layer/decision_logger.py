"""
decision_logger.py

Manages custom database table 'ai_decisions' inside Freqtrade's SQLite database to log model confidence, LLM veto, and risk checks.
"""
import os
import sqlite3
import logging

logger = logging.getLogger(__name__)

class DecisionLogger:
    """
    Handles logging of decision details linked back to Freqtrade's native trade record.
    Schema per §B.7:
    - trade_id: integer / NULL (can link back to Freqtrade's trades table)
    - pair: text
    - timeframe: text
    - tier: integer
    - model_confidence: real
    - di_ok: integer (1 or 0)
    - llm_invoked: integer (1 or 0)
    - llm_provider: text
    - llm_veto: integer (1 or 0)
    - llm_confidence: real
    - llm_reason: text
    - risk_checks_passed: integer (1 or 0)
    - block_reason: text
    - outcome: text
    """
    def __init__(self, db_url: str = "tradesv3.dryrun.sqlite"):
        # Freqtrade database can be specified as a sqlite path or URL
        self.db_path = db_url.replace("sqlite:///", "")
        logger.info(f"Initializing DecisionLogger with SQLite database path: {self.db_path}")
        self.setup_table()

    def setup_table(self):
        """
        Creates the custom ai_decisions table if it does not exist.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    pair TEXT,
                    timeframe TEXT,
                    tier INTEGER,
                    model_confidence REAL,
                    di_ok INTEGER,
                    llm_invoked INTEGER,
                    llm_provider TEXT,
                    llm_veto INTEGER,
                    llm_confidence REAL,
                    llm_reason TEXT,
                    risk_checks_passed INTEGER,
                    block_reason TEXT,
                    outcome TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            conn.close()
            logger.info("Custom 'ai_decisions' table created or already exists.")
        except Exception as e:
            logger.error(f"Failed to setup 'ai_decisions' table in SQLite: {e}")

    def log_decision(self, decision_data: dict):
        """
        Inserts a decision record in the custom ai_decisions table.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_decisions (
                    trade_id, pair, timeframe, tier, model_confidence, di_ok,
                    llm_invoked, llm_provider, llm_veto, llm_confidence, llm_reason,
                    risk_checks_passed, block_reason, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision_data.get("trade_id"),
                decision_data.get("pair"),
                decision_data.get("timeframe"),
                decision_data.get("tier"),
                decision_data.get("model_confidence"),
                decision_data.get("di_ok", 1),
                decision_data.get("llm_invoked", 0),
                decision_data.get("llm_provider", "none"),
                decision_data.get("llm_veto", 0),
                decision_data.get("llm_confidence"),
                decision_data.get("llm_reason"),
                decision_data.get("risk_checks_passed", 1),
                decision_data.get("block_reason"),
                decision_data.get("outcome", "unknown")
            ))
            conn.commit()
            conn.close()
            logger.info(f"Successfully logged AI/Risk decision for {decision_data.get('pair')}")
        except Exception as e:
            logger.error(f"Failed to insert decision record into ai_decisions: {e}")
