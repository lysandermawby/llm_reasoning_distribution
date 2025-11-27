#!/usr/bin/env python3
"""downlading the synthetic-1 dataset"""

# package imports
from datasets import load_dataset # hf download
from pathlib import Path

data_dir = Path("data/")


def main():
    """main script logic"""
    ds = load_dataset("PrimeIntellect/SYNTHETIC-1")
    df = ds.to_pandas()

    df.head()
    

if __name__ == '__main__':
    main()
