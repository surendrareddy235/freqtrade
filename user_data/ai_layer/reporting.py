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
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_reporting")

def parse_last_days(last_str: str) -> int:
    """Parses a duration string like '7days' or '14days' into integer of days."""
    if not last_str:
        return None
    cleaned = last_str.lower().strip()
    if cleaned.endswith("days"):
        num_str = cleaned[:-4]
    elif cleaned.endswith("day"):
        num_str = cleaned[:-3]
    else:
        num_str = cleaned
    try:
        return int(num_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid timeframe format: '{last_str}'. Expected format like '7days', '14days', etc."
        )

def main():
    parser = argparse.ArgumentParser(description="Weekly AI Decisions Review & Calibration Report")
    parser.add_argument("--db-path", type=str, default="user_data/tradesv3.dryrun.sqlite", help="Path to SQLite database")
    parser.add_argument("--last", type=parse_last_days, default=None, help="Timeframe of report (e.g., 7days, 14days)")
    parser.add_argument("--output", type=str, default="console", choices=["console", "csv"], help="Output format")

    args = parser.parse_args()

    # Robust DB path lookup
    db_path = None
    db_candidates = [
        args.db_path,
        "tradesv3.dryrun.sqlite",
        "user_data/tradesv3.dryrun.sqlite",
        "tradesv3.sqlite",
        "user_data/tradesv3.sqlite",
        os.path.join(os.path.dirname(__file__), "..", "..", "user_data", "tradesv3.dryrun.sqlite"),
        os.path.join(os.path.dirname(__file__), "..", "..", "tradesv3.dryrun.sqlite"),
        os.path.join(os.path.dirname(__file__), "..", "..", "user_data", "tradesv3.sqlite"),
        os.path.join(os.path.dirname(__file__), "..", "..", "tradesv3.sqlite"),
    ]

    for candidate in db_candidates:
        if candidate and os.path.exists(candidate):
            db_path = candidate
            break

    if not db_path:
        logger.error(f"Database file not found. Checked candidates: {db_candidates}. Cannot generate report.")
        return

    logger.info(f"Generating report using database: {db_path}...")

    try:
        conn = sqlite3.connect(db_path)

        # Check if ai_decisions table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_decisions'")
        if not cursor.fetchone():
            print("\n--- Weekly AI Decisions & Outcome Report ---")
            print("The 'ai_decisions' table does not exist in the database. No decisions recorded yet.\n")
            conn.close()
            return

        # Check if trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        has_trades_table = cursor.fetchone() is not None

        if has_trades_table:
            query = """
                SELECT
                    d.id as decision_id,
                    d.trade_id,
                    d.timestamp,
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
                    t.is_open,
                    t.close_profit_pct,
                    t.close_date
                FROM ai_decisions d
                LEFT JOIN trades t ON d.trade_id = t.id
                ORDER BY d.timestamp DESC
            """
        else:
            # Fallback if trades table doesn't exist yet
            query = """
                SELECT
                    d.id as decision_id,
                    d.trade_id,
                    d.timestamp,
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
                    NULL as is_open,
                    NULL as close_profit_pct,
                    NULL as close_date
                FROM ai_decisions d
                ORDER BY d.timestamp DESC
            """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            print("\n--- Weekly AI Decisions & Outcome Report ---")
            print("No decisions recorded in the database yet.\n")
            return

        # Convert timestamp to datetime and filter if requested
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')

        if args.last is not None:
            limit_date = datetime.utcnow() - timedelta(days=args.last)
            df = df[df['timestamp_dt'] >= limit_date]

        if df.empty:
            print("\n--- Weekly AI Decisions & Outcome Report ---")
            print(f"No decisions found in the last {args.last} days.\n")
            return

        # Process columns for presentation
        report_rows = []
        for _, row in df.iterrows():
            # ID format: Decision ID / Trade ID
            dec_id = row['decision_id']
            trade_id_val = int(row['trade_id']) if pd.notna(row['trade_id']) else "N/A"
            id_str = f"Dec #{dec_id} / Trade #{trade_id_val}" if trade_id_val != "N/A" else f"Dec #{dec_id}"

            # LLM Veto Verdict
            llm_veto_val = row['llm_veto']
            llm_invoked_val = row['llm_invoked']
            if llm_invoked_val:
                veto_status = "Vetoed" if llm_veto_val else "Approved"
            else:
                veto_status = "N/A"

            # Risk checks passed/failed
            risk_checks_passed_val = row['risk_checks_passed']
            all_passed = False
            if pd.notna(risk_checks_passed_val):
                val_str = str(risk_checks_passed_val).strip()
                if val_str in ["1", "1.0", "True", "true"]:
                    all_passed = True
                elif val_str in ["0", "0.0", "False", "false"]:
                    all_passed = False
                else:
                    try:
                        import json
                        breakdown = json.loads(risk_checks_passed_val)
                        if isinstance(breakdown, dict):
                            all_passed = all(breakdown.values())
                        else:
                            all_passed = bool(breakdown)
                    except Exception:
                        all_passed = bool(risk_checks_passed_val)

            risk_ok = "Passed" if all_passed else "Failed"

            # Block reason if blocked
            block_reason_val = row['block_reason'] if pd.notna(row['block_reason']) and row['block_reason'] else "N/A"

            # Outcome construction:
            # "Win (+x.xx%)", "Loss (-x.xx%)", "Open", "Blocked"
            # For rows vetoed or blocked, show the reason clearly
            if not all_passed:
                outcome_str = f"Blocked ({row['block_reason']})" if row['block_reason'] else "Blocked (Risk check failed)"
            elif llm_invoked_val and llm_veto_val:
                outcome_str = f"Vetoed ({row['llm_reason']})" if row['llm_reason'] else "Vetoed (LLM vetoed)"
            elif pd.notna(row['is_open']):
                if row['is_open'] == 1:
                    outcome_str = "Open"
                else:
                    profit = row['close_profit_pct']
                    if profit is not None:
                        if profit > 0:
                            outcome_str = f"Win (+{profit:.2%})"
                        elif profit < 0:
                            outcome_str = f"Loss ({profit:.2%})"
                        else:
                            outcome_str = "Flat (0.00%)"
                    else:
                        outcome_str = "Closed"
            else:
                decision_outcome_val = row['decision_outcome'] if pd.notna(row['decision_outcome']) else "N/A"
                outcome_str = f"Approved ({decision_outcome_val})"

            report_rows.append({
                "ID": id_str,
                "Timestamp": row['timestamp'],
                "Pair": row['pair'],
                "Tier": row['tier'],
                "Model Conf": f"{row['model_confidence']:.4f}" if pd.notna(row['model_confidence']) else "N/A",
                "LLM Provider": row['llm_provider'] if pd.notna(row['llm_provider']) and row['llm_provider'] != "none" else "N/A",
                "LLM Veto": veto_status,
                "Risk OK": risk_ok,
                "Block Reason": block_reason_val,
                "Outcome": outcome_str
            })

        report_df = pd.DataFrame(report_rows)

        if args.output == "console":
            print("\n" + "="*110)
            print("                       WEEKLY AI DECISIONS & OUTCOME REPORT")
            print("="*110)
            print(tabulate(report_df, headers='keys', tablefmt="grid", showindex=False))
            print("="*110 + "\n")

        elif args.output == "csv":
            csv_path = "ai_decisions_report.csv"
            report_df.to_csv(csv_path, index=False)
            logger.info(f"Report successfully saved to CSV file: {csv_path}")

    except Exception as e:
        logger.error(f"An error occurred while generating report: {e}", exc_info=True)

if __name__ == "__main__":
    main()
