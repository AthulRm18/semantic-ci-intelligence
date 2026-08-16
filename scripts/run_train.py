"""CLI entry point: python scripts/run_train.py"""
import logging

from ci_intel.data.cleaner import clean
from ci_intel.models.train import train_baseline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = clean()
    model, metrics = train_baseline(df)
    print("\nFinal metrics:", metrics)
