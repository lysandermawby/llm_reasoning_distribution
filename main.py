#!/usr/bin/env python3
"""Downloading and processing LLM reasoning data"""

# package imports
import click

# local imports
from dataset_config import datasets_data
import download_utils

# extracting available datasets from the datasets_data config
DATASETS = datasets_data.keys()

@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-l', '--list', is_flag=True, help='show a list of available models')
@click.option('-d', '--dataset', type=click.Choice(DATASETS, case_sensitive=False), required=False, help='download and process a particular dataset')
def main(list, dataset):
    """downloads, processes, and analyses data from reasoning models"""
    if list:
        list_of_datasets = datasets_data.keys()
        print(f"List of datasets which can be downloaded and processed: {', '.join(list_of_datasets)}")

    if dataset:
        df = download_utils.hf_dataset_download(dataset, save=True)
        print(df.head())

if __name__ == "__main__":
    main()
