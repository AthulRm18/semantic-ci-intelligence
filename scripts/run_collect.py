"""CLI entry point: python scripts/run_collect.py"""
import logging

from ci_intel.data.collector import collect

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    collect()
