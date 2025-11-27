#!/usr/bin/env python3
"""Data download and processing functions"""

# package imports
from datasets import load_dataset # hf download
from pathlib import Path

# local imports
from dataset_config import datasets_data
from utils import print_error

def hf_dataset_download(dataset_name, save=True):
    """downloading a dataset from huggingface and return the relevant pandas dataframe"""
    if dataset_name not in datasets_data.keys():
        print_error(f"Error: {dataset_name} is not in {', '.join(datasets_data.keys())}")

    ds_split = datasets_data[dataset_name]

    if ds_split:
        ds = load_dataset(dataset_name, split=ds_split)
    else:
        ds = load_dataset(dataset_name)

    if save:
        data_dir = Path("data")
        dataset_path = data_dir / dataset_name / ".csv" # save this in a directory named after itself
        ds.to_csv(dataset_path)

    # converting to a pandas dataframe
    df = ds.to_pandas()

    return df
