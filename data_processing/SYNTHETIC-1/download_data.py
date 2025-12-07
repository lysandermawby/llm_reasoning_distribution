#!/usr/bin/env python3
"""downlading the synthetic-1 dataset"""

# package imports
from datasets import load_dataset, Dataset # hf download
from pathlib import Path
import click
import json


data_dir = Path("data/")


def download_synthetic(n=10000):
    """download first n elements of the dataset"""
    if not n:
        ds = load_dataset("PrimeIntellect/SYNTHETIC-1", split="train")
    else:
        # Use streaming mode to avoid downloading entire shards
        # This only downloads the data we actually need
        ds_stream = load_dataset(
            "PrimeIntellect/SYNTHETIC-1",
            split="train",
            streaming=True
        )
        # Take only the first n samples as a list
        samples = list(ds_stream.take(n))

        # Clean up the streaming dataset
        del ds_stream

        # Convert to regular dataset
        if samples:
            ds = Dataset.from_dict({
                key: [sample[key] for sample in samples]
                for key in samples[0].keys()
            })
        else:
            # Handle empty case
            ds = Dataset.from_dict({})
    return ds


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option('-n', '--number', type=int, help='number of samples to download', default=100)
@click.option('-a', '--download-all', 'download_all', is_flag=True, help='download all samples')
def main(number, download_all):
    """main script logic"""
    # Validate mutually exclusive options
    if download_all:
        number = None
    
    # Download dataset
    if download_all:
        click.echo("Downloading all samples...")
        ds = download_synthetic(n=None)
    else:
        click.echo(f"Downloading {number} samples...")
        ds = download_synthetic(n=number)

    # Process dataset
    df = ds.to_pandas()
    click.echo(df.head())

    data_path = data_dir / "SYNTHETIC-1.jsonl"
    with open(data_path, "w") as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f)
            f.write('\n')

    print(f"Written {len(df.index)} entries to {data_path}")


if __name__ == '__main__':
    main()
