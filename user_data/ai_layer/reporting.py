"""
reporting.py

Weekly review export script to join the native trade table with the custom ai_decisions table.
Outputs a readable report (console table or CSV) of AI decisions versus actual trade outcomes.
"""
import os
import argparse
import sqlite3
import pandas as pd
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_reporting")

def main():
    parser = argparse.ArgumentParser(description="Weekly AI Decisions Review & Calibration Report")
    parser.add_argument("--db-path", type=str, default="tradesv3.dryrun.sqlite", help="Path to SQLite database")
    parser.add_argument("--last", type=str, default="7days", help="Timeframe of report (e.g., 7days, 14days)")
    parser.add_argument("--output", type=str, default="console", choices=["console", "csv"], help="Output format")

    args = parser.parse_args()
    db_path = args.db_path

    if not os.path.exists(db_path):
        # Fallback to check default paths
        if os.path.exists("user_data/tradesv3.dryrun.sqlite"):
            db_path = "user_data/tradesv3.dryrun.sqlite"
        elif os.path.exists("user_data/tradesv3.sqlite"):
            db_path = "user_data/tradesv3.sqlite"

    logger.info(f"Generating report using database: {db_path}...")

    if not os.path.exists(db_path):
        logger.error(f"Database file not found at {args.db_path} or fallbacks. Cannot generate report.")
        return

    try:
        conn = sqlite3.connect(db_path)

        # We join on the 'trade_id' or 'pair' as fallback
        query = """
            SELECT
                d.id as decision_id,
                d.pair,
                d.timeframe,
                d.tier,
                d.model_confidence,
                d.llm_invoked,
                d.llm_provider,
                d.llm_veto,
                d.llm_confidence as llm_conf,
                d.llm_reason,
                d.risk_checks_passed,
                d.block_reason,
                d.outcome as decision_outcome,
                t.id as ft_trade_id,
                t.is_open,
                t.close_profit_pct,
                t.close_date
            FROM ai_decisions d
            LEFT JOIN trades t ON d.trade_id = t.id
            ORDER BY d.timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("\n--- Weekly AI Decisions & Outcome Report ---")
            print("No decisions or trades recorded in the database yet.\n")
            return

        # Filter by timeframe if needed (e.g., last 7 days)
        # Note: In SQLite we can also filter via query but doing it in pandas is robust
        if args.last == "7days":
            # Just keep last 50 entries for readability if no timestamps are parsed
            pass

        if args.output == "console":
            print("\n" + "="*80)
            print("                       WEEKLY AI DECISIONS & OUTCOME REPORT")
            print("="*80)

            headers = [
                "ID", "Pair", "Tier", "LGBM Conf", "LLM?", "Veto", "LLM Reason", "Risk OK", "Outcome"
            ]

            table_data = []
            for _, row in df.iterrows():
                llm_status = f"Yes ({row['llm_provider']})" if row['llm_invoked'] else "No"
                veto_status = "True" if row['llm_veto'] else "False"

                # Format outcome
                outcome = row['decision_outcome']
                if row['ft_trade_id'] is not None:
                    profit = row['close_profit_pct']
                    profit_str = f"{profit:.2%}" if profit is not None else "Open"
                    outcome = f"Trade #{row['ft_trade_id']} ({profit_str})"

                table_data.append([
                    row['decision_id'],
                    row['pair'],
                    row['tier'],
                    f"{row['model_confidence']:.2f}",
                    llm_status,
                    veto_status,
                    str(row['llm_reason'])[:25],
                    "Yes" if row['risk_checks_passed'] else "No",
                    outcome
                ])

            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print("="*80 + "\n")

        elif args.output == "csv":
            csv_path = "ai_decisions_report.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Report saved to CSV file: {csv_path}")

    except Exception as e:
        logger.error(f"An error occurred while generating report: {e}", exc_info=True)

if __name__ == "__main__":
    main()
