#!/usr/bin/env python3
"""Data download and processing functions"""

# package imports
from datasets import load_dataset, Dataset
# from pathlib import Path
import json
import gc
from tqdm import tqdm


def download_hf_dataset_streaming(
    hf_repo_name, data_dir, output_filename = None, n = None, split = "train", name = None):
    """download a HuggingFace dataset using streaming mode and save as JSONL"""
    # Create data directory
    data_dir.mkdir(exist_ok=True)

    # set the output_filename
    if not output_filename:
        output_filename = hf_repo_name.split("/")[1] + ".jsonl"

    # Download dataset
    if not n:
        # Download entire dataset
        ds = load_dataset(hf_repo_name, name=name, split=split)
    else:
        # Use streaming mode to download only n samples
        ds_stream = load_dataset(
            hf_repo_name,
            name=name,
            split=split,
            streaming=True
        )
        # Collect n samples with progress bar
        samples = list(tqdm(ds_stream.take(n), total=n, desc="Downloading samples"))

        # Delete streaming dataset immediately to stop background threads - If deleted, resource cleanup will not occur correctly
        del ds_stream
        gc.collect()

        # Convert to regular dataset
        ds = Dataset.from_dict({
            key: [sample[key] for sample in samples]
            for key in samples[0].keys()
        }) if samples else Dataset.from_dict({})

    # Convert to pandas and save as JSONL
    df = ds.to_pandas()
    # print(df.head())

    data_path = data_dir / output_filename
    with open(data_path, "w") as f:
        for _, row in df.iterrows():
            # Convert to dict and handle numpy arrays
            row_dict = {}
            for key, value in row.items():
                # Convert numpy arrays to lists for JSON serialization
                if hasattr(value, 'tolist'):
                    row_dict[key] = value.tolist()
                else:
                    row_dict[key] = value
            json.dump(row_dict, f)
            f.write('\n')

    print(f"Written {len(df.index)} entries to {data_path}")
    return data_path
