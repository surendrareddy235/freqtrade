"""
reporting.py

Weekly review export script to join the native trade table with the custom ai_decisions table.
Outputs a readable report (console table or CSV) of AI decisions versus actual trade outcomes.
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_reporting")

def main():
    parser = argparse.ArgumentParser(description="Weekly AI Decisions Review & Calibration Report")
    parser.add_argument("--last", type=str, default="7days", help="Timeframe of report (e.g., 7days, 14days)")
    parser.add_argument("--output", type=str, default="console", choices=["console", "csv"], help="Output format")

    args = parser.parse_args()
    logger.info("Generating report for last %s in %s format...", args.last, args.output)

    # Placeholder for Phase 1
    print("\n--- Weekly AI Decisions & Outcome Report (Phase 1 Placeholder) ---")
    print("No actual trades or AI decisions recorded yet.\n")

if __name__ == "__main__":
    main()
