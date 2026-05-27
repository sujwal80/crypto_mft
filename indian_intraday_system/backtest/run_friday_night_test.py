"""Main script executing the Friday Night Math Test daily backtest."""

from indian_intraday_system.backtest.backtest_runner import BacktestRunner


def main():
    # 1. Initialize Backtester
    runner = BacktestRunner(starting_capital=150000.0)

    # 2. Execute simulated breakout day: NIFTY spot surges from 21,980.00 to 22,180.00
    # This forces a breakout past the Call Wall (22,100) to trigger buy fills and exits.
    runner.run_backtest_day(date_str="2026-05-22", prev_close=21980.0, force_eod_spot=22180.0)

    # 3. Print complete quantitative report
    runner.print_backtest_report()


if __name__ == "__main__":
    main()
