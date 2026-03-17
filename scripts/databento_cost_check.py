"""Check Databento download cost for ES futures 1-min OHLCV data.

Usage:
    export DATABENTO_API_KEY='your-key-here'
    python scripts/databento_cost_check.py
"""

import os

try:
    import databento as db
except ImportError:
    print("Error: 'databento' package not installed. Run: pip install databento")
    raise SystemExit(1)


def main():
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("Error: set DATABENTO_API_KEY environment variable first.")
        print("  export DATABENTO_API_KEY='your-key-here'")
        return

    client = db.Historical(api_key)

    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=["ES.c.0"],
        schema="ohlcv-1m",
        start="2015-01-01",
        end="2024-12-31",
        stype_in="continuous",
    )
    print(f"ES 1-min OHLCV (2015-2024) cost: ${cost:.2f}")


if __name__ == "__main__":
    main()
